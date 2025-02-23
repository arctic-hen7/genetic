use super::ProbabilitySelector;
use crate::population::{Individual, Population};

/// A selector that assigns probabilities of selection to individuals by their rank in the list
/// from worst to best fitnesses, where higher individuals have a better chance of being selected.
/// Unlike a roulette selector, the probabilities are not proportional to the actual fitness
/// scores, only to their relative ordering (meaning the advantage an individual with a fitness of
/// 50 has over one with 49 is the same as the one an individual with fitness 500 has over one with
/// fitness 400 if they're next to each other in the ranking). This can mitigate over-selection
/// with roulette-based systems, though it can also provide insufficient selection pressure.
pub struct RankSelector {
    beta: f64,
}
impl RankSelector {
    /// Creates a new rank selector that applies a uniform selection pressure (i.e. individuals
    /// with a higher fitness are more likely to be chosen, but not more than proportionally so).
    pub fn fair() -> Self {
        Self { beta: 1.0 }
    }
    /// Creates a new rank selector with the given bias toward higher-performing individuals. Beta
    /// greater than 1.0 will lead to a selection pressure whereby the selective advantage of
    /// individuals with higher ranks increases exponentially (proportional to their rank though,
    /// *not* their fitness).
    pub fn new_with_beta(beta: f64) -> Self {
        Self { beta }
    }
}
impl<I: Individual> ProbabilitySelector<I> for RankSelector {
    fn probabilities(&self, population: &Population<I>) -> Vec<f64> {
        // Assign each individual a probability of being chosen proportional to their fitness
        let total_weight = (1..=population.size())
            .map(|i| self.beta.powi(i as i32))
            .sum::<f64>();

        population
            .individuals()
            .enumerate()
            .map(|(idx, _)| (self.beta.powi((idx + 1) as i32)) / total_weight)
            .collect()
    }

    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {}
}
