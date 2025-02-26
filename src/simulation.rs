use rand::{random, Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use crate::{
    error::{SimulationError, SimulationSetupError},
    pair_selector::{NullPairSelector, PairSelector},
    population::{Generator, Individual, Population, SimpleGenerator},
    population_updater::{DirectPopulationUpdater, PopulationUpdater},
    stop_criterion::{
        MaxGenerationsStopCriterion, NullStopCriterion, OrStopCriterion, StopCriterion,
    },
    DefaultRng, IndividualRef, Tracker,
};
use std::{
    any::{Any, TypeId},
    marker::PhantomData,
};

/// A simulation using a genetic algorithm to solve an optimisation problem.
pub struct Simulation<
    I: Individual,
    G: Generator<I>,
    P: PairSelector<I>,
    U: PopulationUpdater<I>,
    S: StopCriterion<I>,
> {
    generator: G,
    pair_selector: P,
    tracker: I::Tracker,
    population_updater: U,
    stop_criterion: S,

    initial_population_size: Option<usize>,
    seed: Option<u64>,

    _ty: PhantomData<I>,
}
impl<
        I: Individual,
        G: Generator<I>,
        P: PairSelector<I> + Any,
        U: PopulationUpdater<I>,
        S: StopCriterion<I> + Any,
    > Simulation<I, G, P, U, S>
{
    /// Runs the simulation.
    ///
    /// Before calling this, a [`PairSelector`] and [`StopCriterion`] must have been set up, along
    /// with an initial population size.
    pub fn run(mut self) -> Result<SimulationResults<I>, SimulationError> {
        // Check that all mandatory fields are set
        let size = self
            .initial_population_size
            .ok_or(SimulationSetupError::NoPopulationSize)?;

        // NOTE: No check for generator, tracker, or population updater, because their defaults are
        // valid
        if self.pair_selector.type_id() == TypeId::of::<NullPairSelector>() {
            return Err(SimulationSetupError::NoPairSelector.into());
        }
        if self.stop_criterion.type_id() == TypeId::of::<NullStopCriterion>() {
            return Err(SimulationSetupError::NoStopCriterion.into());
        }
        // If the user didn't provide a seed, seed randomly
        let seed = self.seed.unwrap_or_else(|| random());
        let mut rng = DefaultRng::seed_from_u64(seed);

        // Generate the population and compute all their fitnesses (generator is responsible for
        // using the RNG deterministically)
        let mut population = Population::new(self.generator.generate(size, &mut rng));

        let mut num_generations = 0;
        loop {
            // Get an iterator that will select the next mating pair in the population (this
            // implicitly does fitness computations, rankings, etc.); this will finish once we have
            // enough offspring. Again, it is the selector's responsibility to handle the RNG
            // deterministically.
            let mut pair_selector_rng = DefaultRng::from_rng(&mut rng);
            let pairs = self
                .pair_selector
                .select(&population, &mut pair_selector_rng);

            // We'll cross/mutate in parallel, seeding each independent block with the same RNG
            // (deterministic)
            let child_seed: u64 = rng.random();
            let new_population = pairs
                // NOTE: This is operating on a *parallel* iterator
                .enumerate()
                .map(|(idx, (parent1, parent2))| {
                    let mut rng = DefaultRng::seed_from_u64(idx as u64 ^ child_seed);

                    // Cross the parents and mutate to produce a child solution
                    let mut child = parent1.cross(&parent2, &self.tracker, &mut rng);
                    child.mutate(&self.tracker, &mut rng);
                    child
                })
                .collect::<Vec<_>>();

            // Update the population (might carry individuals from the old population to the new
            // one). We don't compute the fitnesses of the new population or sort it, or anything
            // really, because there are helper methods that the updater can use to do that (if it
            // didn't need to, that would be a waste).
            let old_individuals = self.population_updater.update_population(
                new_population,
                &mut population,
                &mut rng,
            );

            // Update everyone's cross-generational statistics (population updater and stop
            // criterion will handle it separately)
            self.tracker
                .process_generation(&population, &old_individuals);
            self.pair_selector
                .process_generation(&population, &old_individuals);

            // We've now completed a generation, if we stop here we will have done that many
            num_generations += 1;

            // Check if we've reached a satisfactory solution
            if self
                .stop_criterion
                .should_stop(&population, &old_individuals)
            {
                break;
            }
        }

        // We've found a solution, report the results
        Ok(SimulationResults {
            final_population: population,
            num_generations,
            final_tracker: self.tracker,
            seed,
        })
    }

    /// Seeds the simulation's random number generator (currently [`DefaultRng`], though
    /// this may change in future). This allows simulations to be entirely deterministic, meaning
    /// they will produce the same results.
    ///
    /// If this is not called, a random seed will be used (which cannot later be recovered, so if
    /// you want to be able to replicate a result, you should call this).
    ///
    /// *Note: deterministic evaluation may vary between devices due to floating-point arithmetic
    /// differences, and especially between 32 and 64-bit devices.*
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the initial size of the population. This method must be called at some stage before
    /// [`Self::run`].
    ///
    /// In the majority of algorithms, this will be the constant size of the population, maintained
    /// throughout the simulation, but if the [`PairSelector`] chosen produces a different number
    /// of mating pairs of solutions, the population size may vary, hence this only strictly
    /// controls the initial size.
    pub fn with_initial_population_size(mut self, size: usize) -> Self {
        self.initial_population_size = Some(size);
        self
    }

    /// Sets the [`Generator`] to use for the simulation, which will control how the initial
    /// population is generated. Typically, this doesn't need to be called, because individuals can
    /// be generated independently (the default is [`SimpleGenerator`]), but if your simulation
    /// requires some invariants to be preserved over the population, you'll need to write your own
    /// and use this method.
    ///
    /// Note that custom generators can also be used when solution types are generic over multiple
    /// problems (e.g. a solution type representing a general vector, with the objective function
    /// referenced and provided by a pre-set, problem-specific, generator).
    pub fn with_generator<G2: Generator<I>>(self, generator: G2) -> Simulation<I, G2, P, U, S> {
        Simulation {
            generator,
            pair_selector: self.pair_selector,
            tracker: self.tracker,
            population_updater: self.population_updater,
            stop_criterion: self.stop_criterion,

            initial_population_size: self.initial_population_size,
            seed: self.seed,

            _ty: self._ty,
        }
    }

    /// Sets the [`PairSelector`] to use for the simulation, which will control how mating pairs of
    /// solutions are chosen. This method must be called at some stage before [`Self::run`].
    pub fn with_selector<P2: PairSelector<I>>(
        self,
        pair_selector: P2,
    ) -> Simulation<I, G, P2, U, S> {
        Simulation {
            generator: self.generator,
            pair_selector,
            tracker: self.tracker,
            population_updater: self.population_updater,
            stop_criterion: self.stop_criterion,

            initial_population_size: self.initial_population_size,
            seed: self.seed,

            _ty: self._ty,
        }
    }

    /// Sets the [`PopulationUpdater`] to use for the simulation, which will control how the
    /// population is updated from one generation to the next. This method must be called at some
    /// stage before [`Self::run`].
    pub fn with_updater<U2: PopulationUpdater<I>>(
        self,
        population_updater: U2,
    ) -> Simulation<I, G, P, U2, S> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            tracker: self.tracker,
            population_updater,
            stop_criterion: self.stop_criterion,

            initial_population_size: self.initial_population_size,
            seed: self.seed,

            _ty: self._ty,
        }
    }

    /// Sets the [`StopCriterion`] to use for the simulation, which will control when the
    /// simulation should stop. This method, or [`Self::max_generations`], must be called at some
    /// stage before [`Self::run`].
    ///
    /// Multiple stop criteria can be applied using the [`StopCriterion::or`],
    /// [`StopCriterion::and`], and [`StopCriterion::not`] methods to combine them in logical
    /// chains.
    pub fn with_stop_criterion<S2: StopCriterion<I>>(
        self,
        stop_criterion: S2,
    ) -> Simulation<I, G, P, U, S2> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            tracker: self.tracker,
            population_updater: self.population_updater,
            stop_criterion,

            initial_population_size: self.initial_population_size,
            seed: self.seed,

            _ty: self._ty,
        }
    }
}
impl<
        I: Individual,
        G: Generator<I>,
        P: PairSelector<I> + 'static,
        U: PopulationUpdater<I> + 'static,
    > Simulation<I, G, P, U, NullStopCriterion>
{
    /// Stops the simulation after the given maximum number of generations. If you'd like to use a
    /// different [`StopCriterion`], call [`Self::with_stop_criterion`] and then
    /// [`Self::or_max_generations`] instead (if you still want to limit the number of
    /// generations).
    pub fn max_generations(
        self,
        max_generations: usize,
    ) -> Simulation<I, G, P, U, MaxGenerationsStopCriterion> {
        self.with_stop_criterion(MaxGenerationsStopCriterion::new(max_generations))
    }
}
impl<
        I: Individual,
        G: Generator<I>,
        P: PairSelector<I>,
        U: PopulationUpdater<I>,
        S: StopCriterion<I>,
    > Simulation<I, G, P, U, S>
{
    /// Regardless of what the other stop criterion says, stop the simulation after the given
    /// number of generations has passed. This is equivalent to calling
    /// [`Self::with_stop_criterion`] with your arbitrarily complex stop criterion, and then
    /// calling [`StopCriterion::or`] with a [`MaxGenerationsStopCriterion`].
    pub fn or_max_generations(
        self,
        max_generations: usize,
    ) -> Simulation<I, G, P, U, OrStopCriterion<I, S, MaxGenerationsStopCriterion>> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            tracker: self.tracker,
            population_updater: self.population_updater,
            stop_criterion: self
                .stop_criterion
                .or(MaxGenerationsStopCriterion::new(max_generations)),

            initial_population_size: self.initial_population_size,
            seed: self.seed,

            _ty: self._ty,
        }
    }
}

/// A builder for a [`Simulation`].
///
/// *Note: this actually returns a [`Simulation`] immediately, but not one that will execute
/// without an error. First, a selector, stop criterion, and initial population size must be
/// specified.*
pub struct SimulationBuilder<I: Individual> {
    _ty: PhantomData<I>,
}
impl<I: Individual> SimulationBuilder<I> {
    /// Creates a new simulation. This will require specifying the [`Individual`] being simulated
    /// over, and then the `.with_*` methods can be called in a builder-pattern to set everything
    /// else up generically.
    pub fn new() -> Simulation<
        I,
        SimpleGenerator, // Actually valid
        NullPairSelector,
        DirectPopulationUpdater, // Actually valid
        NullStopCriterion,
    > {
        Simulation {
            generator: SimpleGenerator,
            pair_selector: NullPairSelector,
            tracker: I::tracker(),
            population_updater: DirectPopulationUpdater,
            stop_criterion: NullStopCriterion,

            initial_population_size: None,
            seed: None,

            _ty: PhantomData,
        }
    }
}

/// The final results of a simulation. This allows access to the underlying final [`Population`],
/// which can be used to get the best performing individual in the last generation, to iterate over
/// the individuals in that generation, etc. It also exposes generic simulation statistics.
pub struct SimulationResults<I: Individual> {
    final_population: Population<I>,
    final_tracker: I::Tracker,
    num_generations: usize,
    seed: u64,
}
impl<I: Individual> SimulationResults<I> {
    /// Returns the number of generations that occurred before the simulation terminated.
    pub fn num_generations(&self) -> usize {
        self.num_generations
    }

    /// Returns the seed used, which allows reproducing this result.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the final value of the cross-generational tracker associated with the
    /// individuals being simulated over.
    pub fn last_tracker(&self) -> &I::Tracker {
        &self.final_tracker
    }

    /// Returns the best individual (in the last generation, and, if the simulation is properly
    /// configured, hopefully ever) in the simulation. This is `None` if there were no individuals
    /// in the last generation (impossible with constant-size population simulations).
    pub fn best(&self) -> Option<IndividualRef<I>> {
        self.final_population.best_individual()
    }

    /// Gets the last generation of the population in this simulation. This allows accessing all
    /// the individuals and qualities of that generation.
    pub fn last_generation(&self) -> &Population<I> {
        &self.final_population
    }
}
