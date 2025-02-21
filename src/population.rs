use crate::crosser::Crosser;
use crate::mutator::{Parameter, ParameterMutation};
use generic_array::{ArrayLength, GenericArray};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};
use rayon::slice::ParallelSlice;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Deref;

/// An individual in a genetic algorithm.
pub trait Individual: Clone + PartialEq + Eq + Hash + Send + Sync + 'static {
    /// The type of the fitness value of an individual, which must be able to be compared with the
    /// fitnesses of others (ideally very efficiently). Typically, this will be an [`f64`] or
    /// similar.
    type Fitness: Ord + Copy + Send + Sync;
    /// The number of parameters this individual has. This is used in working with arrays of
    /// [`Parameter`]s.
    type N: ArrayLength;
    /// The crosser this individual uses. See [`Crosser`] for details. Most of the time, this will
    /// be [`NullCrosser`] if your simulation's crossing behaviour does not need to change as it
    /// progresses.
    type Crosser: Crosser<Self>;

    /// Generates a random individual.
    ///
    /// Note: if population-level invariants should be preserved during the generation process, a
    /// custom [`Generator`] may be specified at the simulation level and this can be left
    /// unimplemented.
    fn generate() -> Self;

    /// Produces an instance of the crosser.
    fn crosser() -> Self::Crosser;

    /// Computes the *fitness* of the individual, which will be maximised by the genetic algorithm
    /// to converge on the best possible solution. Typically, this is the objective function being
    /// maximised.
    ///
    /// This function must be entirely deterministic, and its return value will be cached to avoid
    /// multiple evaluations.
    ///
    /// Note: if you find yourself specifying a differentiable function here and the fitness
    /// landscape isn't too noisy, you can probably use gradient descent and get better results!
    fn fitness(&self) -> Self::Fitness;

    /// Crosses this individual with another, producing a new individual that is a combination of
    /// the two "parents". This takes a reference to this individual's associated crosser, which is
    /// used to track crossing-related statistics or metrics across generations, allowing crossing
    /// behaviour to change as a simulation progresses.
    fn cross(&self, other: &Self, crosser: &Self::Crosser) -> Self;

    /// Mutates this individual from the given shape of a mutation.
    fn mutate(&mut self, shape: GenericArray<ParameterMutation, Self::N>);

    /// Returns this shape of the individual so mutators can generate a mutation for it without
    /// knowing further details. See [`MutationShape`].
    fn shape() -> GenericArray<Parameter, Self::N>;
}

/// A generator that produces a population of individuals, which is used to start a simulation.
pub trait Generator<I: Individual> {
    /// Generates the given number of individuals.
    fn generate(&self, size: usize) -> Vec<I>;
}

/// A [`Generator`] for any individual that generates a population of the given size by generating
/// that many individuals, without caring about relationships between them. Most simulations will
/// use this, though if you want to ensure initial population diversity or similar, especially in
/// small populations, you should implement a custom generator.
pub struct SimpleGenerator;
impl<I: Individual> Generator<I> for SimpleGenerator {
    fn generate(&self, size: usize) -> Vec<I> {
        (0..size).map(|_| I::generate()).collect()
    }
}

/// A cache of individuals' fitnesses.
pub(crate) struct FitnessCache<I: Individual> {
    /// A cache of fitness values of individuals that have appeared in the population at any time,
    /// which ensures we never need to compute the fitness for an individual twice.
    // TODO: Make this an LRU cache or similar so memory use doesn't blow up
    inner: HashMap<I, I::Fitness>,
}
impl<I: Individual> FitnessCache<I> {
    /// Creates a new fitness cache, starting with the given individuals. Calling this with the
    /// initial population is more efficient that calling it with an empty slice and then calling
    /// [`Self::register_all`], because this knows the cache is empty and skips additional checks
    /// (which ends up saving a whole $O(n)$ loop).
    pub(crate) fn new(individuals: &[I]) -> Self {
        let mut cache = HashMap::new();
        cache.par_extend(
            individuals
                .as_parallel_slice()
                .par_iter()
                .map(|i| (i.clone(), i.fitness())),
        );

        Self { inner: cache }
    }

    /// Gets the given individual's fitness from the cache, if it's there. This will not compute
    /// the fitness if it's not in the cache.
    fn get_fitness(&self, individual: &I) -> Option<I::Fitness> {
        self.inner.get(individual).copied()
    }

    /// Computes the given individual's fitness, fetching it from the cache first or computing it
    /// manually and adding it to the cache if it's not there.
    pub(crate) fn fitness(&mut self, individual: &I) -> I::Fitness {
        if let Some(fitness) = self.inner.get(individual) {
            *fitness
        } else {
            let fitness = individual.fitness();
            self.inner.insert(individual.clone(), fitness.clone());
            fitness
        }
    }

    /// Registers all the given individuals in this fitness cache. This will be a heavy work
    /// function.
    pub(crate) fn register_all(&mut self, individuals: &[I]) {
        // Work out which indices in the slice of individuals have already been registered (when
        // this is executed on a new population, this should be fairly small); we collect these
        // separately to avoid trying to extend `self.inner` from an iterator based on filtering by
        // a reference to `self.inner`
        let skip_indices = individuals
            .iter()
            .filter(|i| self.inner.contains_key(i))
            .enumerate()
            .map(|(idx, _)| idx)
            .collect::<HashSet<_>>();

        // Then actually compute the fitnesses in parallel and extend
        let iter = individuals
            .as_parallel_slice()
            .par_iter()
            .enumerate()
            .filter(|(idx, _)| !skip_indices.contains(idx))
            .map(|(_, i)| (i.clone(), i.fitness()));
        self.inner.par_extend(iter);
    }
}

// pub struct Population<I: Individual> {
//     /// The current population of individuals.
//     population: Vec<I>,
//     /// Whether or not the population is ordered by fitness, to prevent multiple reorderings of the
//     /// same list.
//     is_ordered: bool,
// }
// impl<I: Individual> Population<I> {
//     pub fn new(population: Vec<Individual<N>>, fitness_fn: FitnessFn<F, N>) -> Self {
//         Self {
//             population,
//             is_ordered: false,
//             fitness_fn,
//             fitness_cache: HashMap::new(),
//         }
//     }
//
//     /// Replaces the current population with a new one (which is assumed to be unordered by fitness).
//     pub fn replace_population(&mut self, new_population: Vec<Individual<N>>) {
//         self.population = new_population;
//         self.is_ordered = false;
//     }
//
//     /// Returns the given individual's fitness, fetching it from the cache if possible or computing
//     /// it if not.
//     pub fn fitness(&mut self, individual: &Individual<N>) -> F {
//         Self::fitness_from_cache(individual, &mut self.fitness_cache, &self.fitness_fn)
//     }
//
//     /// Orders the population by fitness, if needed (if it hasn't been changed since it was last
//     /// sorted, this will do nothing).
//     pub fn sort_by_fitness(&mut self) {
//         if !self.is_ordered {
//             self.population.sort_by_key(|individual| {
//                 self.fitness_cache
//                     .get(individual)
//                     .copied()
//                     .unwrap_or_else(|| (self.fitness_fn)(individual))
//             });
//         }
//     }
//
//     /// Converts the population into a sorted list of individuals (by fitness) with their
//     /// fitnesses.
//     pub fn into_sorted_with_fitness(mut self) -> Vec<(Individual<N>, F)> {
//         self.sort_by_fitness();
//         self.population
//             .into_iter()
//             .map(|individual| (self.fitness_cache[&individual], individual))
//             .map(|(f, i)| (i, f))
//             .collect()
//     }
//
//     /// Gets the individual with the greatest fitness, returning their fitness as well. This will
//     /// return [`None`] if there are no individuals in the population. If there are multiple
//     /// best individuals, this will return the first one encountered (which should be assumed to be
//     /// a random one, as other parts of the system may reorder the underlying population
//     /// arbitrarily).
//     ///
//     /// This takes a mutable reference because it computes the fitnesses of individuals in the
//     /// population, and thereby mutates the cache.
//     pub fn best_individual(&mut self) -> Option<(&Individual<N>, F)> {
//         let mut best = None;
//         for individual in &self.population {
//             let fitness =
//                 Self::fitness_from_cache(individual, &mut self.fitness_cache, &self.fitness_fn);
//             if best.map_or(true, |(_, best_fitness)| fitness > best_fitness) {
//                 best = Some((individual, fitness))
//             }
//         }
//
//         best
//     }
// }

/// A container for the population in a simulation, including a cache of the fitnesses of all
/// individuals. References to this will be passed to different parts of the system, and they can
/// be used to access the best-performing individuals, iterate over fitnesses, etc.
pub struct Population<I: Individual> {
    /// The current population of individuals, ordered from least to greatest fitness.
    population: Vec<I>,
    /// A cache of the fitnesses of the individuals this population has seen, which is guaranteed
    /// to include all those in `self.population`.
    fitness_cache: FitnessCache<I>,
}
impl<I: Individual> Population<I> {
    /// Creates a new population handler for the given list of individuals. Behind the scenes, this
    /// will generate a fitness cache and compute the fitness for all provided individuals in
    /// parallel.
    pub(crate) fn new(mut population: Vec<I>) -> Self {
        let fitness_cache = FitnessCache::new(&population);

        // Sort the population from lowest fitness to highest (we can't invert `I::Fitness` easily,
        // so sorting from highest to lowest would be another O(n) loop, which we can avoid by just
        // pre-reversing the iterator whenever we need it)
        population.sort_by_key(|i| fitness_cache.get_fitness(i).unwrap());

        Self {
            population,
            fitness_cache,
        }
    }

    /// Returns the number of individuals currently in the population.
    pub fn size(&self) -> usize {
        self.population.len()
    }

    /// Gets an iterator of all the individuals in the population, sorted by their fitness. Each
    /// individual's fitness is precomputed, and can be fetched with the shadow method
    /// [`IndividualRef::fitness`].
    ///
    /// This iterator is double-ended, and can be reversed to produce the individuals in order of
    /// fitness from worst to best.
    pub fn individuals(&self) -> impl DoubleEndedIterator<Item = IndividualRef<'_, '_, I>> {
        self.population
            .iter()
            // We have the individuals sorted from worst to best, but produce them best to worst by
            // default, and the user can reverse this again if needed
            .rev()
            .map(|individual| IndividualRef::new(individual, &self.fitness_cache))
    }

    /// Returns the best individual in the population, if the population has any individuals at
    /// all.
    pub fn best_individual(&self) -> Option<IndividualRef<'_, '_, I>> {
        self.population
            .first()
            .map(|individual| IndividualRef::new(individual, &self.fitness_cache))
    }

    /// Gets the fitness of the given individual, computing it and registering it in the cache if it
    /// wasn't already in there. In general, only population updaters will need to use this method
    /// for individuals in a potential next generation (i.e. not already in this population
    /// container). Anyone else should use [`Self::individuals`] and similar methods (which return
    /// [`IndividualRef`]s that have fitnesses accessible without potential expensive computation).
    ///
    /// As this may compute the underlying fitness if it hasn't been already, calling more than one
    /// of these should be done in a parallel iterator using [`rayon`] wherever possible.
    pub fn fitness(&mut self, individual: &I) -> I::Fitness {
        self.fitness_cache.fitness(individual)
    }

    /// Replaces the population with the given one, ensuring the fitnesses of all individuals are
    /// computed and that the population is sorted by fitness.
    ///
    /// The simplest [`crate::population_updater::PopulationUpdater`] would just call this method
    /// with the new individuals directly, with no additional work. Most of the time, however, the
    /// fitness of at least some of the individuals in the new population will need to be computed
    /// ahead of time, in which case [`Self::register_and_sort_other`] may be helpful.
    pub fn replace(&mut self, mut new_individuals: Vec<I>) -> Vec<I> {
        self.fitness_cache.register_all(&new_individuals);
        new_individuals.sort_by_key(|i| self.fitness_cache.get_fitness(i).unwrap());

        std::mem::replace(&mut self.population, new_individuals)
    }

    /// Same as [`Self::replace`], but skips all checks that the given new individuals have all
    /// been registered in the fitness cache and are sorted from worst fitness to best. If these
    /// invariants are not upheld, panics will happen later down the line, and [`Self::replace`]
    /// should be called instead.
    pub fn replace_with_sorted_and_registered(&mut self, new_individuals: Vec<I>) -> Vec<I> {
        debug_assert!(new_individuals
            .iter()
            .all(|i| self.fitness_cache.get_fitness(i).is_some()));

        std::mem::replace(&mut self.population, new_individuals)
    }

    /// Computes the fitness of all the given individuals in a potential new population and
    /// registers them in the cache, sorting those individuals from worst fitness to best
    /// (**important:** unles [`Self::individuals`], if you want the best individual first, you'll
    /// need to reverse an iterator produced by `new_individuals`!).
    ///
    /// This does not replace the current population with these new individuals, it just prepares
    /// for that to happen, while allowing comparison between these new individuals and the
    /// previous generation. As such, this is a very common method for population updaters to call.
    pub fn register_and_sort_other(&mut self, new_individuals: &mut Vec<I>) {
        self.fitness_cache.register_all(&new_individuals);
        new_individuals.sort_by_key(|i| self.fitness_cache.get_fitness(i).unwrap());
    }
}

/// A reference to an [`Individual`] that shadows the fitness function to avoid recomputing it
/// unnecessarily. This dereferences to the underlying individual, so can be used as if it were
/// that.
pub struct IndividualRef<'i, 'f, I: Individual> {
    /// The individual we're referencing.
    inner: &'i I,
    /// A fitness cache, which *must* include the fitness of the individual we're referencing.
    fitness_cache: &'f FitnessCache<I>,
}
impl<'i, 'f, I: Individual> IndividualRef<'i, 'f, I> {
    /// Creates a new reference to the given individual, using the given fitness cache.
    pub(crate) fn new(inner: &'i I, fitness_cache: &'f FitnessCache<I>) -> Self {
        debug_assert!(fitness_cache.get_fitness(inner).is_some());

        Self {
            inner,
            fitness_cache,
        }
    }
    /// Gets the fitness of the individual from the cache.
    pub fn fitness(&self) -> I::Fitness {
        self.fitness_cache.get_fitness(self.inner).unwrap()
    }
}
impl<'i, 'f, I: Individual> Deref for IndividualRef<'i, 'f, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}
