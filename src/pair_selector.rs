use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};

use crate::population::{Individual, IndividualRef, Population};

/// A trait for systems that select mating pairs from a population. In a genetic algorithm, these
/// play the role of environmental circumstances, deciding which solutions are "good enough" to
/// produce child solutions. In nearly all cases, this will rely on some kind of ranking by a
/// metric of fitness, though this can be arbitrarily complex.
pub trait PairSelector<I: Individual + 'static> {
    /// Selects the pairs of solutions which will be crossed. The same individual(s) can appear
    /// multiple times.
    ///
    /// This method must produce exactly the number of pairs needed to produce the next generation
    /// of the population, which is typically the same as the initial population size.
    ///
    /// This method will be called at the start of every generation, so cross-generational
    /// statistics may be tracked that way.
    ///
    /// This method is provided a random number generator, and implementors should be careful to
    /// use it deterministically (it will be seeded deterministically by the caller, but `rayon`
    /// does not always split tasks deterministically, and work-stealing can be problematic, so
    /// patterns like assigning generator to each individual, or ideally a number to each
    /// individual, can help).
    fn select<'p>(
        &mut self,
        population: &'p Population<I>,
        rng: &mut impl Rng,
    ) -> impl ParallelIterator<Item = (IndividualRef<'p, I>, IndividualRef<'p, I>)>;

    /// This method is called at the *end* of each generation, to allow a selector to track
    /// cross-generational statistics. It is provided with the new population, along with the
    /// previous generation.
    ///
    /// All members of the previous generation will still have their fitnesses tracked by the new
    /// [`Population`], so [`Population::fitness`] can be used cheaply to get their fitnesses
    /// (without needing any additional computation, so parallelism is *not* necessary).
    /// Additionally, the old individuals will be sorted *from worst fitness to best*.
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>);
}

/// A pair selector that selects the same individual twice. Because pair selectors are mandatory,
/// if this is used in a final simulation, an error will occur.
pub struct NullPairSelector;
impl<I: Individual + 'static> PairSelector<I> for NullPairSelector {
    fn select<'p>(
        &mut self,
        _population: &'p Population<I>,
        _rng: &mut impl Rng,
    ) -> impl ParallelIterator<Item = (IndividualRef<'p, I>, IndividualRef<'p, I>)> {
        unimplemented!();

        // Need this for type hints
        #[allow(unreachable_code)]
        std::iter::empty().par_bridge().into_par_iter()
    }
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {
        unimplemented!()
    }
}
