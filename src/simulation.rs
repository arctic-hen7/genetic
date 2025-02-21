use crate::{
    crosser::{Crosser, NullCrosser},
    error::{SimulationError, SimulationSetupError},
    mutator::{Mutator, NullMutator},
    pair_selector::{NullPairSelector, PairSelector},
    population::{Generator, Individual, Population, SimpleGenerator},
    population_updater::{DirectPopulationUpdater, PopulationUpdater},
    stop_criterion::{
        MaxGenerationsStopCriterion, NullStopCriterion, OrStopCriterion, StopCriterion,
    },
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
    M: Mutator<I>,
    U: PopulationUpdater<I>,
    S: StopCriterion<I>,
> {
    generator: G,
    pair_selector: P,
    crosser: I::Crosser,
    mutator: M,
    population_updater: U,
    stop_criterion: S,

    initial_population_size: Option<usize>,

    _ty: PhantomData<I>,
}
impl<
        I: Individual,
        G: Generator<I>,
        P: PairSelector<I> + Any,
        M: Mutator<I> + Any,
        U: PopulationUpdater<I>,
        S: StopCriterion<I> + Any,
    > Simulation<I, G, P, M, U, S>
{
    /// Creates a new simulation. This will require specifying the [`Individual`] being simulated
    /// over, and then the `.with_*` methods can be called in a builder-pattern to set everything
    /// else up generically.
    pub fn new<I1: Individual>() -> Simulation<
        I1,
        SimpleGenerator, // Actually valid
        NullPairSelector,
        NullMutator,
        DirectPopulationUpdater, // Actually valid
        NullStopCriterion,
    > {
        Simulation {
            generator: SimpleGenerator,
            pair_selector: NullPairSelector,
            crosser: I1::crosser(),
            mutator: NullMutator,
            population_updater: DirectPopulationUpdater,
            stop_criterion: NullStopCriterion,

            initial_population_size: None,

            _ty: PhantomData,
        }
    }

    /// Runs the simulation.
    ///
    /// Before calling this, a [`PairSelector`] and [`Mutator`] must have been set up, along with
    /// some [`StopCriterion`].
    pub fn run(mut self) -> Result<Vec<(I, I::Fitness)>, SimulationError> {
        // Check that all mandatory fields are set
        let size = self
            .initial_population_size
            .ok_or(SimulationSetupError::NoPopulationSize)?;

        // NOTE: No check for generator or population updater, because their defaults are valid
        if self.pair_selector.type_id() == TypeId::of::<NullPairSelector>() {
            return Err(SimulationSetupError::NoPairSelector.into());
        }
        if self.crosser.type_id() == TypeId::of::<NullCrosser>() {
            return Err(SimulationSetupError::NoCrosser.into());
        }
        if self.mutator.type_id() == TypeId::of::<NullMutator>() {
            return Err(SimulationSetupError::NoMutator.into());
        }
        if self.stop_criterion.type_id() == TypeId::of::<NullStopCriterion>() {
            return Err(SimulationSetupError::NoStopCriterion.into());
        }

        // Generate the population and compute all their fitnesses
        let mut population = Population::new(self.generator.generate(size));

        let i_shape = I::shape();
        loop {
            // Get an iterator that will select the next mating pair in the population (this
            // implicitly does fitness computations, rankings, etc.); this will finish once we have
            // enough offspring
            let pairs = self.pair_selector.select(&population);
            let mut new_population = Vec::new();
            for (parent1, parent2) in pairs {
                // Cross the parents and mutate to produce a child solution
                let mut child = parent1.cross(&parent2, &self.crosser);
                let mutation_pattern = self.mutator.mutate(&i_shape);
                child.mutate(mutation_pattern);
                new_population.push(child);
            }

            // Update the population (might carry individuals from the old population to the new
            // one). We don't compute the fitnesses of the new population or sort it, or anything
            // really, because there are helper methods that the updater can use to do that (if it
            // didn't need to, that would be a waste).
            let old_individuals = self
                .population_updater
                .update_population(new_population, &mut population);

            // Update everyone's cross-generational statistics (population updater and stop
            // criterion will handle it separately)
            self.crosser
                .process_generation(&population, &old_individuals);
            self.pair_selector
                .process_generation(&population, &old_individuals);
            self.mutator
                .process_generation(&population, &old_individuals);

            // Check if we've reached a satisfactory solution
            if self
                .stop_criterion
                .should_stop(&population, &old_individuals)
            {
                break;
            }
        }

        // We've found a solution, report the final individuals and their fitnesses, in order
        Ok(population
            .individuals()
            .map(|i| (i.clone(), i.fitness()))
            .collect())
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

    /// Sets the [`PairSelector`] to use for the simulation, which will control how mating pairs of
    /// solutions are chosen. This method must be called at some stage before [`Self::run`].
    pub fn with_selector<P2: PairSelector<I>>(
        self,
        pair_selector: P2,
    ) -> Simulation<I, G, P2, M, U, S> {
        Simulation {
            generator: self.generator,
            pair_selector,
            crosser: self.crosser,
            mutator: self.mutator,
            population_updater: self.population_updater,
            stop_criterion: self.stop_criterion,

            initial_population_size: self.initial_population_size,

            _ty: self._ty,
        }
    }

    /// Sets the [`Mutator`] to use for the simulation, which will control how solutions are
    /// mutated. This method must be called at some stage before [`Self::run`].
    pub fn with_mutator<M2: Mutator<I>>(self, mutator: M2) -> Simulation<I, G, P, M2, U, S> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            crosser: self.crosser,
            mutator,
            population_updater: self.population_updater,
            stop_criterion: self.stop_criterion,

            initial_population_size: self.initial_population_size,

            _ty: self._ty,
        }
    }

    /// Sets the [`PopulationUpdater`] to use for the simulation, which will control how the
    /// population is updated from one generation to the next. This method must be called at some
    /// stage before [`Self::run`].
    pub fn with_updater<U2: PopulationUpdater<I>>(
        self,
        population_updater: U2,
    ) -> Simulation<I, G, P, M, U2, S> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            crosser: self.crosser,
            mutator: self.mutator,
            population_updater,
            stop_criterion: self.stop_criterion,

            initial_population_size: self.initial_population_size,

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
    ) -> Simulation<I, G, P, M, U, S2> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            crosser: self.crosser,
            mutator: self.mutator,
            population_updater: self.population_updater,
            stop_criterion,

            initial_population_size: self.initial_population_size,

            _ty: self._ty,
        }
    }
}
impl<
        I: Individual,
        G: Generator<I>,
        P: PairSelector<I> + 'static,
        M: Mutator<I> + 'static,
        U: PopulationUpdater<I> + 'static,
    > Simulation<I, G, P, M, U, NullStopCriterion>
{
    /// Stops the simulation after the given maximum number of generations. If you'd like to use a
    /// different [`StopCriterion`], call [`Self::with_stop_criterion`] and then
    /// [`Self::or_max_generations`] instead (if you still want to limit the number of
    /// generations).
    pub fn max_generations(
        self,
        max_generations: usize,
    ) -> Simulation<I, G, P, M, U, MaxGenerationsStopCriterion> {
        self.with_stop_criterion(MaxGenerationsStopCriterion::new(max_generations))
    }
}
impl<
        I: Individual,
        G: Generator<I>,
        P: PairSelector<I>,
        M: Mutator<I>,
        U: PopulationUpdater<I>,
        S: StopCriterion<I>,
    > Simulation<I, G, P, M, U, S>
{
    /// Regardless of what the other stop criterion says, stop the simulation after the given
    /// number of generations has passed. This is equivalent to calling
    /// [`Self::with_stop_criterion`] with your arbitrarily complex stop criterion, and then
    /// calling [`StopCriterion::or`] with a [`MaxGenerationsStopCriterion`].
    pub fn or_max_generations(
        self,
        max_generations: usize,
    ) -> Simulation<I, G, P, M, U, OrStopCriterion<I, S, MaxGenerationsStopCriterion>> {
        Simulation {
            generator: self.generator,
            pair_selector: self.pair_selector,
            crosser: self.crosser,
            mutator: self.mutator,
            population_updater: self.population_updater,
            stop_criterion: self
                .stop_criterion
                .or(MaxGenerationsStopCriterion::new(max_generations)),

            initial_population_size: self.initial_population_size,

            _ty: self._ty,
        }
    }
}
