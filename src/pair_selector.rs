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
    fn select(
        &mut self,
        population: &Population<I>,
    ) -> impl Iterator<Item = (IndividualRef<'_, '_, I>, IndividualRef<'_, '_, I>)>;

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
pub(crate) struct NullPairSelector;
impl<I: Individual + 'static> PairSelector<I> for NullPairSelector {
    fn select(
        &mut self,
        _population: &Population<I>,
    ) -> impl Iterator<Item = (IndividualRef<'_, '_, I>, IndividualRef<'_, '_, I>)> {
        unimplemented!();

        // Need this for type hints
        #[allow(unreachable_code)]
        std::iter::empty()
    }
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {
        unimplemented!()
    }
}
