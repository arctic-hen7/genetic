use crate::{
    population::{Individual, IndividualRef, Population},
    DefaultRng, PairSelector,
};
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// A tournament-based selector that picks a random subset of the population each time (making
/// `tournament_size` draws) and then picks the two best individuals from that subset.
///
/// This selector maintains no cross-generational state, and will run the tournament in the same
/// way in every generation.
pub struct TournamentSelector {
    tournament_size: usize,
}
impl TournamentSelector {
    /// Creates a new tournament selector with the given size. A larger size will take longer to
    /// evaluate, but will produce better results more quickly.
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}
impl<I: Individual> PairSelector<I> for TournamentSelector {
    fn select<'p>(
        &self,
        population: &'p Population<I>,
        rng: &mut impl Rng,
    ) -> impl IndexedParallelIterator<Item = (IndividualRef<'p, I>, IndividualRef<'p, I>)> + 'p
    {
        let tournament_size = self.tournament_size;
        (0..population.size())
            .into_iter()
            .map(|_| DefaultRng::from_rng(rng))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(move |mut rng| {
                // Choose `tournament_size` indices; because they're already ordered from worst to
                // best, the two highest indices win and mate
                let mut best_index = 0;
                let mut second_best_index = 0;

                for _ in 0..tournament_size {
                    let idx = rng.random_range(0..population.size());
                    if idx > best_index {
                        second_best_index = best_index;
                        best_index = idx;
                    } else if idx > second_best_index {
                        second_best_index = idx;
                    }
                }

                (
                    population.individual_at(best_index).unwrap(),
                    population.individual_at(second_best_index).unwrap(),
                )
            })
    }

    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {}
}
