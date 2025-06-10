use crate::{
    population::{Individual, IndividualRef, Population},
    DefaultRng, PairSelector,
};
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[derive(Clone)]
pub struct ProbabilitySelectorWrapper<I: Individual, S: ProbabilitySelector<I>> {
    inner: S,
    _ty: std::marker::PhantomData<I>,
}

/// A simple "meta-selector" designed for making it easier to implement selectors which first
/// assign a probability to all individuals in the population, and then select based on those
/// probabilities. This does the selecting part, given a function that assigns probabilities.
///
/// Due to Rust's orphan rule, before using a probability selector, you'll need to call the
/// provided method `.wrap()` on it.
pub trait ProbabilitySelector<I: Individual> {
    /// Assigns probabilities to all individuals in the population, in order from worst to best.
    fn probabilities(&self, population: &Population<I>) -> Vec<f64>;
    /// Processes the next generation. See [`PairSelector::process_generation`] for details.
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>);

    /// Wraps this selector in a [`ProbabilitySelectorWrapper`], which implements [`PairSelector`].
    fn wrap(self) -> ProbabilitySelectorWrapper<I, Self>
    where
        Self: Sized,
    {
        ProbabilitySelectorWrapper {
            inner: self,
            _ty: std::marker::PhantomData,
        }
    }
}
impl<I: Individual, S: ProbabilitySelector<I>> PairSelector<I>
    for ProbabilitySelectorWrapper<I, S>
{
    fn select<'p>(
        &self,
        population: &'p Population<I>,
        rng: &mut impl Rng,
    ) -> impl IndexedParallelIterator<Item = (IndividualRef<'p, I>, IndividualRef<'p, I>)> + 'p
    {
        // Assign each individual a probability of being chosen
        let mut probabilities = self.inner.probabilities(population);
        debug_assert!(!probabilities.iter().any(|p| p != p),);
        // Convert those to cumulative probabilities
        let mut total_so_far = 0.0;
        for p in &mut probabilities {
            *p += total_so_far;
            total_so_far = *p;
        }

        let global_seed: u64 = rng.random();
        (0..population.size()).into_par_iter().enumerate().map_with(
            // Need to take ownership of `probabilities` in each thread
            probabilities,
            move |probabilities, (idx, _)| {
                let mut rng = DefaultRng::seed_from_u64(idx as u64 ^ global_seed);

                // Use a binary search to find the first individual with a cumulative probability
                // greater than or equal to a random number
                let min_cum_prob: f64 = rng.random_range(0.0..=1.0);
                // Panic on any NaNs or similar, but whether we get an exact match or the index at
                // which we could insert what we want, go with whatever's there (in the latter
                // case, we'd get the index of the next highest probability)
                let first_idx = probabilities
                    .binary_search_by(|&p| p.partial_cmp(&min_cum_prob).unwrap())
                    .map_or_else(|idx| idx, |idx| idx);

                // Do that again for the other parent
                let min_cum_prob: f64 = rng.random_range(0.0..=1.0);
                let second_idx = probabilities
                    .binary_search_by(|&p| p.partial_cmp(&min_cum_prob).unwrap())
                    .map_or_else(|idx| idx, |idx| idx);

                (
                    population.individual_at(first_idx).unwrap(),
                    population.individual_at(second_idx).unwrap(),
                )
            },
        )
    }

    // Delegate generation processing (i.e. state updating) to the sub-selector
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>) {
        self.inner
            .process_generation(population, previous_generation);
    }
}
