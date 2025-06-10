use crate::population::{Individual, IndividualRef, Population};
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};

/// A trait for systems that select mating pairs from a population. In a genetic algorithm, these
/// play the role of environmental circumstances, deciding which solutions are "good enough" to
/// produce child solutions. In nearly all cases, this will rely on some kind of ranking by a
/// metric of fitness, though this can be arbitrarily complex.
pub trait PairSelector<I: Individual> {
    /// Selects the pairs of solutions which will be crossed. The same individual(s) can appear
    /// multiple times.
    ///
    /// This method must produce exactly the number of pairs needed to produce the next generation
    /// of the population, which is typically the same as the initial population size.
    ///
    /// This method is provided a random number generator, and implementors should be careful to
    /// use it deterministically (it will be seeded deterministically by the caller, but `rayon`
    /// does not always split tasks deterministically, and work-stealing can be problematic, so
    /// patterns like assigning generator to each individual, or ideally a number to each
    /// individual, can help).
    fn select<'p>(
        &self,
        population: &'p Population<I>,
        rng: &mut impl Rng,
    ) -> impl IndexedParallelIterator<Item = (IndividualRef<'p, I>, IndividualRef<'p, I>)> + 'p;

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

/// A pair selector that never selects any individuals. Because pair selectors are mandatory, if
/// this is used in a final simulation, an error will occur.
#[derive(Clone, Copy)]
pub struct NullPairSelector;
impl<I: Individual> PairSelector<I> for NullPairSelector {
    fn select<'p>(
        &self,
        _population: &'p Population<I>,
        _rng: &mut impl Rng,
    ) -> impl IndexedParallelIterator<Item = (IndividualRef<'p, I>, IndividualRef<'p, I>)> + 'p
    {
        unimplemented!("you must specify a real pair selector for your simulation");

        // Need this for type hints
        #[allow(unreachable_code)]
        Vec::new().into_par_iter()
    }
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {
        unimplemented!()
    }
}
