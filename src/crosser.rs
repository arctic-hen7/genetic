use crate::population::{Individual, Population};

pub trait Crosser<I: Individual> {
    /// Crosses the two given individuals to produce a new individual.
    fn cross(&self, a: &I, b: &I) -> I;

    /// This method is called at the *end* of each generation, to allow a crosser to track
    /// cross-generational statistics. It is provided with the new population, along with the
    /// previous generation.
    ///
    /// All members of the previous generation will still have their fitnesses tracked by the new
    /// [`Population`], so [`Population::fitness`] can be used cheaply to get their fitnesses
    /// (without needing any additional computation, so parallelism is *not* necessary).
    /// Additionally, the old individuals will be sorted *from worst fitness to best*.
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>);
}

/// A crosser that does nothing. Because crossers are mandatory, if this is used in a final
/// simulation, an error will occur.
pub(crate) struct NullCrosser;
impl<I: Individual> Crosser<I> for NullCrosser {
    fn cross(&self, _: &I, _: &I) -> I {
        unimplemented!()
    }
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {
        unimplemented!()
    }
}
