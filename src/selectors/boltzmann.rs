use super::ProbabilitySelector;
use crate::population::{Individual, Population};
use std::{
    f64::consts::E,
    iter::Sum,
    ops::{Div, Sub},
};

/// A Boltzmann selector, which assigns probabilities of selection to individuals based on their
/// fitness, exponentially scaled, and adjusted by a temperature parameter that can vary between
/// generations. The way the temperature changes can be customised over generations, though most
/// methods will require a known maximal number of generations.
pub struct BoltzmannSelector<I: Individual> {
    /// The current temperature.
    temperature: f64,
    /// Function that changes the temperature on each generation, taking in the temperature from
    /// the previous generation, the new population, and a vector of individuals from the previous
    /// generation. See [`ProbabilitySelector::process_generation`] for details. This should return
    /// the temperature for the next generation.
    temp_changer: Box<dyn FnMut(f64, &Population<I>, &Vec<I>) -> f64>,
}
impl<I: Individual> BoltzmannSelector<I> {
    /// Creates a new custom [`BoltzmannSelector`] with the given initial temperature and function
    /// that changes the temperature between generations.
    pub fn new_custom(
        initial_temp: f64,
        temp_changer: impl FnMut(f64, &Population<I>, &Vec<I>) -> f64 + 'static,
    ) -> Self {
        Self {
            temperature: initial_temp,
            temp_changer: Box::new(temp_changer),
        }
    }
    /// Creates a new [`BoltzmannSelector`] where the temperature decreases *linearly* from the
    /// given initial temperature to the given final temperature, over the given number of
    /// generations. If the simulation goes for any longer than this number of generations, the
    /// behaviour of this selector may be nonsensical (e.g. negative probabilities, divisions by
    /// zero, etc.).
    pub fn new_linear(initial_temp: f64, final_temp: f64, max_generations: usize) -> Self {
        let delta = (initial_temp - final_temp) / max_generations as f64;

        Self::new_custom(initial_temp, move |temp, _, _| temp - delta)
    }
    /// Creates a new [`BoltzmannSelector`] where the temperature decreases *exponentially* from
    /// the given initial temperature, with the given decay rate. This can run for arbitrarily many
    /// generations, though eventually floating-point imprecision will lead to unpredictable
    /// behaviour.
    pub fn new_exponential(initial_temp: f64, decay_rate: f64) -> Self {
        Self::new_custom(initial_temp, move |temp, _, _| temp * decay_rate)
    }
    /// Creates a new [`BoltzmannSelector`] where the temperature decreases *sigmoidally*, i.e.
    /// decreasing slowly at first, then more rapidly, and then slowly again. The slope parameter
    /// controls the general slope of the curve (higher means more rapid selection in the middle),
    /// and the `pivot_generation` is the index of the generation where the temperature will be
    /// decreasing most rapidly. Generally, this should be `max_generations / 2`, but it may be
    /// useful in some simulations to adjust this.
    pub fn new_sigmoidal(initial_temp: f64, slope: f64, pivot_generation: usize) -> Self {
        let mut gen = 0isize;
        Self::new_custom(initial_temp, move |_, _, _| {
            gen += 1;
            initial_temp / (1.0 + E.powf(slope * ((gen - pivot_generation as isize) as f64)))
        })
    }
}
impl<I: Individual> ProbabilitySelector<I> for BoltzmannSelector<I>
where
    I::Fitness: Sum + Div<f64, Output = f64> + Sub<Output = I::Fitness> + std::fmt::Debug,
{
    fn probabilities(&self, population: &Population<I>) -> Vec<f64> {
        debug_assert!(self.temperature != 0.0);

        // To avoid exponents overflowing, we'll normalise all fitnesses by subtracting the minimum
        // (this has no mathematical effect)
        let min_fitness = population.individuals().next().unwrap().fitness();
        let prob_sum = population
            .individuals()
            .map(|i| E.powf((i.fitness() - min_fitness) / self.temperature))
            .sum::<f64>();

        population
            .individuals()
            .map(|i| E.powf((i.fitness() - min_fitness) / self.temperature) / prob_sum)
            .collect()
    }

    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>) {
        self.temperature = (self.temp_changer)(self.temperature, population, previous_generation);
    }
}
