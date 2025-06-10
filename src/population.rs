use ahash::RandomState as ARandomState;
use lru::LruCache;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::collections::HashSet;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ops::Deref;

use crate::Tracker;

/// An individual in a genetic algorithm.
pub trait Individual: Clone + PartialEq + Eq + Hash + Send + Sync {
    /// The type of the fitness value of an individual, which must be able to be compared with the
    /// fitnesses of others (ideally very efficiently). Typically, this will be an [`f64`] or
    /// similar.
    type Fitness: Ord + Copy + Send + Sync;
    /// The tracker this individual uses. See [`Tracker`] for details. Most of the time, this will
    /// be [`NullTracker`] if your simulation's crossing/mutation behaviour does not need to change
    /// as the simulation progresses.
    type Tracker: Tracker<Self> + Send + Sync;

    /// Generates a random individual.
    ///
    /// Note: if population-level invariants should be preserved during the generation process, a
    /// custom [`Generator`] may be specified at the simulation level and this can be left
    /// unimplemented.
    fn generate(rng: &mut impl Rng) -> Self;

    /// Produces an instance of the tracker. This will be called once, at the start of the
    /// simulation.
    fn tracker() -> Self::Tracker;

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
    /// the two "parents". This takes a reference to the tracker we've instantiated to follow the
    /// simulation and compute cross-generational statistics, which allows modifying crossing
    /// behaviour as the simulation progresses.
    fn cross(&self, other: &Self, tracker: &Self::Tracker, rng: &mut impl Rng) -> Self;

    /// Mutates this individual in-place to maintain diversity in the population.
    ///
    /// It's very common to use the provided [`Tracker`] here to do things like alter the frequency
    /// of mutations as the simulation goes on to balance between "exploration" of the solution
    /// space and "exploitation" of the good solutions we've found.
    fn mutate(&mut self, tracker: &Self::Tracker, rng: &mut impl Rng);
}

/// A generator that produces a population of individuals, which is used to start a simulation.
pub trait Generator<I: Individual> {
    /// Generates the given number of individuals.
    ///
    /// This method is provided a random number generator, and implementors should be careful to
    /// use it deterministically (it will be seeded deterministically by the caller, but `rayon`
    /// does not always split tasks deterministically, and work-stealing can be problematic, so
    /// patterns like assigning generator to each individual, or ideally a number to each
    /// individual, can help). In general, generation will be done in sequence, but in the rare
    /// cases where it isn't, bear this in mind.
    fn generate(&self, size: usize, rng: &mut impl Rng) -> Vec<I>;
}

/// A [`Generator`] for any individual that generates a population of the given size by generating
/// that many individuals, without caring about relationships between them. Most simulations will
/// use this, though if you want to ensure initial population diversity or similar, especially in
/// small populations, you should implement a custom generator.
pub struct SimpleGenerator;
impl<I: Individual> Generator<I> for SimpleGenerator {
    fn generate(&self, size: usize, rng: &mut impl Rng) -> Vec<I> {
        // Sequential generation to ensure deterministic RNG use
        (0..size).map(|_| I::generate(rng)).collect()
    }
}

/// A cache of individuals' fitnesses.
pub(crate) struct FitnessCache<I: Individual> {
    /// A cache of fitness values of individuals that have appeared in the population at any time,
    /// which ensures we never need to compute the fitness for an individual twice.
    inner: LruCache<I, I::Fitness, ARandomState>,
}
impl<I: Individual> FitnessCache<I> {
    /// Creates a new fitness cache, starting with the given individuals. Calling this with the
    /// initial population is more efficient that calling it with an empty slice and then calling
    /// [`Self::register_all`], because this knows the cache is empty and skips additional checks
    /// (which ends up saving a whole $O(n)$ loop).
    ///
    /// This will compute the fitness of all the given individuals in parallel.
    ///
    /// # Panics
    ///
    /// This will panic if the given cache size if zero.
    pub(crate) fn new(individuals: &[I], size: usize) -> Self {
        let mut cache = LruCache::with_hasher(
            NonZeroUsize::new(size).expect("fitness cache size must be non-zero"),
            ARandomState::default(),
        );
        let kv_pairs = individuals
            .as_parallel_slice()
            .par_iter()
            .map(|i| (i.clone(), i.fitness()))
            .collect_vec_list();
        for list in kv_pairs {
            for (k, v) in list {
                cache.push(k, v);
            }
        }

        Self { inner: cache }
    }

    /// Gets the given individual's fitness from the cache, if it's there. This will not compute
    /// the fitness if it's not in the cache.
    ///
    /// Note: under the hood, we store fitnesses using an LRU cache (meaning the last recently used
    /// will be evicted). This method does *not* update the usage of the element you're trying to
    /// access!
    fn try_fitness(&self, individual: &I) -> Option<I::Fitness> {
        self.inner.peek(individual).copied()
    }

    /// Computes the given individual's fitness, fetching it from the cache first or computing it
    /// manually and adding it to the cache if it's not there.
    pub(crate) fn fitness(&mut self, individual: &I) -> I::Fitness {
        if let Some(fitness) = self.inner.get(individual) {
            *fitness
        } else {
            let fitness = individual.fitness();
            self.inner.push(individual.clone(), fitness.clone());
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
            .par_iter()
            .enumerate()
            .filter(|(_, i)| self.inner.contains(i))
            .map(|(idx, _)| idx)
            .collect::<HashSet<_, ARandomState>>();
        // We must promote all those individuals to avoid them getting lost! If that happened,
        // individuals we register may not have their fitnesses *actually* registered
        for idx in &skip_indices {
            self.inner.promote(&individuals[*idx]);
        }

        // Then actually compute the fitnesses in parallel and extend
        let kv_pairs = individuals
            .as_parallel_slice()
            .par_iter()
            .enumerate()
            .filter(|(idx, _)| !skip_indices.contains(idx))
            .map(|(_, i)| (i.clone(), i.fitness()))
            .collect_vec_list();
        for list in kv_pairs {
            for (k, v) in list {
                self.inner.push(k, v);
            }
        }
    }
}

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
        // NOTE: Right now, we're just allowing 50% more space than one generation takes up. Might
        // change this in future or allow the user to?
        let fitness_cache = FitnessCache::new(&population, population.len() * 3 / 2);

        // Sort the population from lowest fitness to highest (we can't invert `I::Fitness` easily,
        // so sorting from highest to lowest would be another O(n) loop, which we can avoid by just
        // pre-reversing the iterator whenever we need it)
        population.par_sort_unstable_by_key(|i| fitness_cache.try_fitness(i).unwrap());

        Self {
            population,
            fitness_cache,
        }
    }

    /// Returns the number of individuals currently in the population.
    pub fn size(&self) -> usize {
        self.population.len()
    }

    /// Gets an iterator of all the individuals in the population, sorted by their fitness (worst
    /// to best). Each individual's fitness is precomputed, and can be fetched with the shadow
    /// method [`IndividualRef::fitness`].
    ///
    /// This iterator is double-ended, and can be reversed to produce the individuals in order of
    /// fitness from best to worst.
    pub fn individuals(&self) -> impl DoubleEndedIterator<Item = IndividualRef<'_, I>> {
        self.population
            .iter()
            .map(|individual| IndividualRef::new(individual, &self.fitness_cache))
    }

    /// Gets the individual at the given position in the list of individuals, ordered from worst to
    /// best by their fitnesses.
    pub fn individual_at(&self, idx: usize) -> Option<IndividualRef<'_, I>> {
        self.population
            .get(idx)
            .map(|individual| IndividualRef::new(individual, &self.fitness_cache))
    }

    /// Returns the best individual in the population, if the population has any individuals at
    /// all.
    pub fn best_individual(&self) -> Option<IndividualRef<'_, I>> {
        self.population
            .last()
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

        new_individuals.par_sort_unstable_by_key(|i| self.fitness_cache.try_fitness(i).unwrap());

        std::mem::replace(&mut self.population, new_individuals)
    }

    /// Same as [`Self::replace`], but skips all checks that the given new individuals have all
    /// been registered in the fitness cache and are sorted from worst fitness to best. If these
    /// invariants are not upheld, panics will happen later down the line, and [`Self::replace`]
    /// should be called instead.
    pub fn replace_with_sorted_and_registered(&mut self, new_individuals: Vec<I>) -> Vec<I> {
        debug_assert!(new_individuals
            .iter()
            .all(|i| self.fitness_cache.try_fitness(i).is_some()));

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
        new_individuals.par_sort_unstable_by_key(|i| self.fitness_cache.try_fitness(i).unwrap());
    }
}

/// A reference to an [`Individual`] that shadows the fitness function to avoid recomputing it
/// unnecessarily. This dereferences to the underlying individual, so can be used as if it were
/// that.
pub struct IndividualRef<'a, I: Individual> {
    /// The individual we're referencing.
    inner: &'a I,
    /// A fitness cache, which *should* include the fitness of the individual we're referencing.
    fitness_cache: &'a FitnessCache<I>,
}
impl<'a, I: Individual> IndividualRef<'a, I> {
    /// Creates a new reference to the given individual, using the given fitness cache.
    pub(crate) fn new(inner: &'a I, fitness_cache: &'a FitnessCache<I>) -> Self {
        debug_assert!(fitness_cache.try_fitness(inner).is_some());

        Self {
            inner,
            fitness_cache,
        }
    }
    /// Gets the fitness of the individual from the cache. This will not compute the fitness if
    /// it's already stored (but holding onto an individual reference for more than one generation
    /// may mean its fitness is evicted from the cache, so this will compute it again if
    /// necessary).
    ///
    /// Note: if this function needs to recompute an individual's fitness, it will *not* add that
    /// fitness back to the main cache, so don't expect it to if you use this in a hot loop!
    pub fn fitness(&self) -> I::Fitness {
        self.fitness_cache
            .try_fitness(self.inner)
            .unwrap_or_else(|| self.inner.fitness())
    }
    /// Extracts the inner individual by cloning it.
    pub fn into_inner(&self) -> I {
        self.inner.clone()
    }
}
impl<'a, I: Individual> Deref for IndividualRef<'a, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}
