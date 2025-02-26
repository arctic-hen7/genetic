//! A collection of basic optimisation tests based on fairly simple continuous optimisation
//! problems.

use genetic::selectors::*;
use genetic::*;
use ordered_float::NotNan;

/// An `N`-dimensional ral-valued vector representing a candidate solution to a continuous
/// optimisation problem, using arithmetic crossing and random perturbation mutation strategies.
/// This is incapable of being directly generated, it needs to be generated through
/// [`ObjectiveFnGenerator`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Solution<const N: usize> {
    /// The actual vector of coordinates this solution represents.
    inner: [NotNan<f64>; N],
    /// A pointer to the objective function being *maximised*.
    objective_fn: fn(&[NotNan<f64>; N]) -> NotNan<f64>,
}
impl<const N: usize> Individual for Solution<N> {
    type Fitness = NotNan<f64>;
    type Tracker = NullTracker;

    fn generate(_rng: &mut impl rand::Rng) -> Self {
        unimplemented!()
    }
    fn tracker() -> Self::Tracker {
        NullTracker
    }

    fn fitness(&self) -> Self::Fitness {
        (self.objective_fn)(&self.inner)
    }

    // Arithmetic crossing for continuous optimisation
    fn cross(&self, other: &Self, _tracker: &Self::Tracker, rng: &mut impl rand::Rng) -> Self {
        let alpha = NotNan::new(rng.random_range(0.0..1.0)).unwrap();
        let mut child = [NotNan::new(0.0).unwrap(); N];
        for i in 0..N {
            child[i] = self.inner[i] * alpha + other.inner[i] * (NotNan::new(1.0).unwrap() - alpha);
        }

        Self {
            inner: child,
            objective_fn: self.objective_fn,
        }
    }

    // Random perturbation mutations
    fn mutate(&mut self, _tracker: &Self::Tracker, rng: &mut impl rand::Rng) {
        for i in 0..N {
            // First as if we should mutate, then actually mutate
            if rng.random_bool(MUTATION_PROBABILITY) {
                self.inner[i] += NotNan::new(rng.random_range(-1.0..1.0)).unwrap();
            }
        }
    }
}

/// A [`Generator`] for [`Solution`]s, which generates them by filling their coordinates with
/// random values and setting their objective functions to be what this is instantiated with
/// (allows using the same individual type for many different problems).
struct ObjectiveFnGenerator<const N: usize> {
    objective_fn: fn(&[NotNan<f64>; N]) -> NotNan<f64>,
}
impl<const N: usize> ObjectiveFnGenerator<N> {
    fn new(objective_fn: fn(&[NotNan<f64>; N]) -> NotNan<f64>) -> Self {
        Self { objective_fn }
    }
}
impl<const N: usize> Generator<Solution<N>> for ObjectiveFnGenerator<N> {
    fn generate(&self, size: usize, rng: &mut impl rand::Rng) -> Vec<Solution<N>> {
        (0..size)
            .into_iter()
            .map(|_| {
                let mut indiv = [0.0; N];
                rng.fill(&mut indiv);
                let indiv = indiv.map(|x| NotNan::new(x).unwrap());

                Solution {
                    inner: indiv,
                    objective_fn: self.objective_fn,
                }
            })
            .collect()
    }
}

const MUTATION_PROBABILITY: f64 = 0.3;

// *Could* spuriously fail once or twice with really bad random numbers
#[test]
fn should_minimise_hypersphere() {
    const N: usize = 10;
    const POPULATION_SIZE: usize = 3000;

    // The objective function is the sum of the squares of the coordinates
    let objective_fn = |coords: &[NotNan<f64>; N]| {
        NotNan::new(1.0).unwrap()
            / coords
                .iter()
                .map(|x| NotNan::new(x.powi(2)).unwrap())
                .sum::<NotNan<f64>>()
    };

    let solutions = SimulationBuilder::<Solution<N>>::new()
        .with_generator(ObjectiveFnGenerator::new(objective_fn))
        .with_initial_population_size(POPULATION_SIZE)
        .with_selector(RouletteSelector.wrap())
        .with_updater(StaticElitistUpdater::new_worst(10))
        .max_generations(100)
        .run()
        .unwrap();

    let best_fitness = 1.0 / *solutions.best().unwrap().fitness();
    println!(
        "Best solution: {:?}",
        solutions.best().unwrap().into_inner().inner
    );
    println!("Fitness: {}", best_fitness);
    println!("Achieved in {} generations", solutions.num_generations());
    println!("Seed: {}", solutions.seed());

    if best_fitness.abs() > 1e-3 {
        panic!("best fitness was {best_fitness}, expected at least 1e-3");
    }
}
