use crate::population::{Individual, Population};
use std::{marker::PhantomData, ops::Sub};

/// A stop criterion for a genetic algorithm, which tells the algorithm when to stop (typically
/// when a satisfactory solution has been reached).
pub trait StopCriterion<I: Individual> {
    /// Returns whether or not the algorithm should stop based on the new state of its population
    /// after a generational update, along with the individuals from the previous generation.
    ///
    /// This will be called at the end of each generation, and hence can be used to track the
    /// number of generations that have occurred.
    ///
    /// All members of the previous generation will still have their fitnesses tracked by the new
    /// [`Population`], so [`Population::fitness`] can be used cheaply to get their fitnesses
    /// (without needing any additional computation, so parallelism is *not* necessary).
    /// Additionally, the old members will be sorted *from worst fitness to best*.
    ///
    /// Note that stop criteria generally don't need random number generation, though if yours
    /// does, you can always maintain your own internally.
    fn should_stop(&mut self, population: &Population<I>, old_individuals: &Vec<I>) -> bool;

    /// Combines this stop criterion with another one in a logical OR (the resulting criterion will
    /// tell the algorithm to stop when either this or the other one does).
    fn or<S2: StopCriterion<I>>(self, rhs: S2) -> OrStopCriterion<I, Self, S2>
    where
        Self: Sized,
    {
        OrStopCriterion::new(self, rhs)
    }
    /// Combines this stop criterion with another one in a logical AND (the resulting criterion
    /// will tell the algorithm to stop when both this and the other one do).
    fn and<S2: StopCriterion<I>>(self, rhs: S2) -> AndStopCriterion<I, Self, S2>
    where
        Self: Sized,
    {
        AndStopCriterion::new(self, rhs)
    }
    /// Negates this stop criterion (the resulting criterion will tell the algorithm to stop when
    /// this one says to continue).
    fn not(self) -> NotStopCriterion<I, Self>
    where
        Self: Sized,
    {
        NotStopCriterion::new(self)
    }
}

/// A stop criterion that never stops. Because stop criteria are mandatory, if this is used in a
/// final simulation, an error will occur.
pub struct NullStopCriterion;
impl<I: Individual> StopCriterion<I> for NullStopCriterion {
    fn should_stop(&mut self, _: &Population<I>, _: &Vec<I>) -> bool {
        unimplemented!("you must specify a real stop criterion for your simulation");
    }
}

/// A stop criterion that imposes a maximum number of generations on the genetic algorithm,
/// terminating it after that threshold is reached (i.e. after the fitness of the best individual
/// is greater than or equal to it).
pub struct MaxGenerationsStopCriterion {
    max_generations: usize,
    current_generation: usize,
}
impl MaxGenerationsStopCriterion {
    pub fn new(max_generations: usize) -> Self {
        Self {
            max_generations,
            current_generation: 0,
        }
    }
}
impl<I: Individual> StopCriterion<I> for MaxGenerationsStopCriterion {
    fn should_stop(&mut self, _: &Population<I>, _: &Vec<I>) -> bool {
        self.current_generation += 1;
        self.current_generation >= self.max_generations
    }
}

/// A stop criterion that halts the algorithm once some individual reaches a certain minimum
/// fitness threshold. This should be used in cases where a certain minimum fitness is required of
/// a solution, past which point further optimisation is unnecessary.
pub struct FitnessThresholdStopCriterion<I: Individual> {
    threshold: I::Fitness,
}
impl<I: Individual> FitnessThresholdStopCriterion<I> {
    pub fn new(threshold: I::Fitness) -> Self {
        Self { threshold }
    }
}
impl<I: Individual> StopCriterion<I> for FitnessThresholdStopCriterion<I> {
    fn should_stop(&mut self, population: &Population<I>, _: &Vec<I>) -> bool {
        let best = population.best_individual();
        if let Some(best) = best {
            best.fitness() >= self.threshold
        } else {
            // There are no individuals in the population, so we can't produce any more; not
            // stopping here if we're the only criterion would lead to an infinite loop
            true
        }
    }
}

/// A stop criterion that stops the algorithm once the fitness of the best individual is rising by
/// less than a certain amount over a given window of generations. Note that this does *not* take
/// into account population diversity, so if you're getting stuck in a local optimum for a while
/// before breaking out, this criterion will probably keep you there.
pub struct FitnessPlateauStopCriterion<I: Individual>
where
    I::Fitness: Sub<Output = I::Fitness>,
{
    /// The number of generations to consider the change over.
    window_size: usize,
    /// The threshold of the slope from the best individual from `window_size` generations ago to
    /// the best individual from the current generation, below which we consider the algorithm to
    /// be plateauing.
    slope_threshold: <I::Fitness as Sub>::Output,
    /// The number of times we must be consecutively below the threshold before we stop.
    num_below_threshold: usize,

    /// The best fitnesses from the last `window_size` generations, in order from oldest to newest.
    best_fitnesses: Vec<I::Fitness>,
    /// The number of times we've been below the threshold in a row.
    num_below_threshold_counter: usize,
}
impl<I: Individual> FitnessPlateauStopCriterion<I>
where
    I::Fitness: Sub<Output = I::Fitness>,
{
    /// Creates a new [`FitnessPlateauStopCriterion`] with the given window size (how many
    /// generations to consider), slope threshold (the minimum improvement in the best individual's
    /// fitness over that window before we think the algorithm is plateauing), and the number of
    /// consecutive times we much be below that threshold before we stop.
    pub fn new(
        window_size: usize,
        slope_threshold: <I::Fitness as Sub>::Output,
        num_below_threshold: usize,
    ) -> Self {
        Self {
            window_size,
            slope_threshold,
            num_below_threshold,
            best_fitnesses: Vec::with_capacity(window_size),
            num_below_threshold_counter: 0,
        }
    }
}
impl<I: Individual> StopCriterion<I> for FitnessPlateauStopCriterion<I>
where
    I::Fitness: Sub<Output = I::Fitness>,
{
    fn should_stop(&mut self, population: &Population<I>, _old_individuals: &Vec<I>) -> bool {
        // If the new generation is empty, terminate immediately
        if population.size() == 0 {
            return true;
        }

        // Add the new best fitness to our tracker
        let new_best = population.best_individual().unwrap().fitness();
        self.best_fitnesses.push(new_best.clone());
        // And remove the last one if needed to keep the window size
        if self.best_fitnesses.len() > self.window_size {
            self.best_fitnesses.remove(0);

            // Compute the secant slope of the best fitnesses over the window
            let slope = self.best_fitnesses.last().unwrap().clone()
                - self.best_fitnesses.first().unwrap().clone();

            // If the slope is less than the threshold, record that as a breach
            if slope < self.slope_threshold {
                self.num_below_threshold_counter += 1;
                // And if we've had too many, stop
                self.num_below_threshold_counter >= self.num_below_threshold
            } else {
                // Otherwise, reset the counter and keep going
                self.num_below_threshold_counter = 0;

                false
            }
        } else {
            // If we haven't reached the window size yet, we don't have enough data to know if we
            // should stop, so keep going
            false
        }
    }
}

/// A generic stop criterion that stops the algorithm when *both* the constituent stop criteria say
/// to do so.
pub struct AndStopCriterion<I: Individual, S1: StopCriterion<I>, S2: StopCriterion<I>> {
    s1: S1,
    s2: S2,
    _ty: PhantomData<I>,
}
impl<I: Individual, S1: StopCriterion<I>, S2: StopCriterion<I>> AndStopCriterion<I, S1, S2> {
    pub fn new(s1: S1, s2: S2) -> Self {
        Self {
            s1,
            s2,
            _ty: PhantomData,
        }
    }
}
impl<I: Individual, S1: StopCriterion<I>, S2: StopCriterion<I>> StopCriterion<I>
    for AndStopCriterion<I, S1, S2>
{
    fn should_stop(&mut self, population: &Population<I>, old_individuals: &Vec<I>) -> bool {
        self.s1.should_stop(population, old_individuals)
            && self.s2.should_stop(population, old_individuals)
    }
}

/// A generic stop criterion that stops the algorithm when *either* of the constituent stop
/// criteria say to do so.
pub struct OrStopCriterion<I: Individual, S1: StopCriterion<I>, S2: StopCriterion<I>> {
    s1: S1,
    s2: S2,
    _ty: PhantomData<I>,
}
impl<I: Individual, S1: StopCriterion<I>, S2: StopCriterion<I>> OrStopCriterion<I, S1, S2> {
    pub fn new(s1: S1, s2: S2) -> Self {
        Self {
            s1,
            s2,
            _ty: PhantomData,
        }
    }
}
impl<I: Individual, S1: StopCriterion<I>, S2: StopCriterion<I>> StopCriterion<I>
    for OrStopCriterion<I, S1, S2>
{
    fn should_stop(&mut self, population: &Population<I>, old_individuals: &Vec<I>) -> bool {
        self.s1.should_stop(population, old_individuals)
            || self.s2.should_stop(population, old_individuals)
    }
}

/// A generic stop criterion that stops the algorithm when the constituent stop criterion says to
/// continue.
pub struct NotStopCriterion<I: Individual, S: StopCriterion<I>> {
    s: S,
    _ty: PhantomData<I>,
}
impl<I: Individual, S: StopCriterion<I>> NotStopCriterion<I, S> {
    pub fn new(s: S) -> Self {
        Self {
            s,
            _ty: PhantomData,
        }
    }
}
impl<I: Individual, S: StopCriterion<I>> StopCriterion<I> for NotStopCriterion<I, S> {
    fn should_stop(&mut self, population: &Population<I>, old_individuals: &Vec<I>) -> bool {
        !self.s.should_stop(population, old_individuals)
    }
}
