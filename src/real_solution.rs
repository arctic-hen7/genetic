use crate::{Generator, Individual, NullTracker};
use core::f64;
use ordered_float::NotNan;
use std::ops::Range;

/// A real-valued solution vector with `N` dimensions. This is a common solution type, and as such
/// is implemented by default, and it can also be used as a reference implementation of
/// [`Individual`]. This solution type is designed to be generic over the problem being solved, and
/// so is generated through a [`RealSolutionGenerator`], through which the objective function and
/// other parameters may be specified.
///
/// This won't be a good fit for all real-valued optimisation problems. Specifically, it will always
/// perform arithmetic crossing, and mutation is perturbation-based within a given range. If these
/// are a bad fit for your problem, you shoudl implement your own [`Individual`].
#[derive(Clone)]
pub struct RealSolution<const N: usize> {
    /// The actual vector of coordinates this solution represents.
    pub inner: [NotNan<f64>; N],
    /// The generator used to create this solution, which contains a reference to the objective
    /// function (for evaluating fitness) and the mutation probability.
    generator: &'static RealSolutionGenerator<N>,
}
impl<const N: usize> std::fmt::Debug for RealSolution<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealSolution")
            .field("inner", &self.inner)
            .finish()
    }
}
impl<const N: usize> PartialEq for RealSolution<N> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl<const N: usize> Eq for RealSolution<N> {}
impl<const N: usize> PartialOrd for RealSolution<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fitness().partial_cmp(&other.fitness())
    }
}
impl<const N: usize> Ord for RealSolution<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fitness().cmp(&other.fitness())
    }
}
impl<const N: usize> std::hash::Hash for RealSolution<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}
impl<const N: usize> Individual for RealSolution<N> {
    type Fitness = NotNan<f64>;
    type Tracker = NullTracker;

    fn generate(_rng: &mut impl rand::Rng) -> Self {
        unimplemented!("you need to use `.with_generator(RealSolutionGenerator::new(..))` to generate `RealSolution`s")
    }
    fn tracker() -> Self::Tracker {
        NullTracker
    }

    fn fitness(&self) -> Self::Fitness {
        (self.generator.objective_fn)(&self.inner)
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
            generator: self.generator,
        }
    }

    // Random perturbation mutations
    fn mutate(&mut self, _tracker: &Self::Tracker, rng: &mut impl rand::Rng) {
        for i in 0..N {
            // First as if we should mutate, then actually mutate
            if rng.random_bool(self.generator.mutation_probability) {
                self.inner[i] +=
                    NotNan::new(rng.random_range(self.generator.mutation_range.clone())).unwrap();
            }
        }
    }
}

/// A [`Generator`] for the generic [`RealSolution`] type which allows specifying parameters such
/// as the objective function being optimised.
///
/// # Note
///
/// This is somewhat of an abuse of the point of generators. Typically, a custom generator would be
/// used to preserve invariants over the initial population, but here we make the generator a
/// static reference, and provide it by reference to every individual, meaning they can call the
/// fitness function we derive here, and access other parameters. Typically, these would be
/// individual-specific, but because [`RealSolution`] is designed for arbitrary problems, we
/// specify them through a generator instance. This pattern is usually a good idea for libraries,
/// but not for applications (just hardcode into the [`Individual`] instead).
pub struct RealSolutionGenerator<const N: usize> {
    objective_fn: Box<dyn Fn(&[NotNan<f64>; N]) -> NotNan<f64> + Send + Sync + 'static>,
    mutation_probability: f64,
    mutation_range: Range<f64>,
    generation_range: Range<f64>,
}
impl<const N: usize> RealSolutionGenerator<N> {
    /// Creates a new [`RealSolutionGenerator`] for the problem specified by the given objective
    /// function. By default, this function will be *maximised*, though specifying `minimise =
    /// true` will cause it to be minimised instead. The mutation probability is the probability
    /// any given component of the solution vector will be mutated within `mutation_range`.
    ///
    /// The given objective function is guaranteed to never receive a `NaN`, and it should never
    /// return one. If it does, the simulation will panic.
    ///
    /// Note that, if minimising, fitness values will be taken as `1.0 / x`, where `x` is the
    /// objective function value. As a result, if you need to display fitness values later, you
    /// should reciprocate them again:
    ///
    /// ```rust
    /// let best_fitness = 1.0 / *solutions.best().unwrap().fitness();
    /// ```
    pub fn new(
        objective_fn: impl Fn([f64; N]) -> f64 + Send + Sync + 'static,
        minimise: bool,
        mutation_probability: f64,
        mutation_range: Range<f64>,
        generation_range: Range<f64>,
    ) -> &'static Self {
        Box::leak(Box::new(Self {
            objective_fn: if minimise {
                Box::new(move |x| NotNan::new(1.0 / objective_fn(x.map(|i| *i))).unwrap())
            } else {
                Box::new(move |x| NotNan::new(objective_fn(x.map(|i| *i))).unwrap())
            },
            mutation_probability,
            mutation_range,
            generation_range,
        }))
    }
}
impl<const N: usize> Generator<RealSolution<N>> for &'static RealSolutionGenerator<N> {
    fn generate(&self, size: usize, rng: &mut impl rand::Rng) -> Vec<RealSolution<N>> {
        (0..size)
            .into_iter()
            .map(|_| {
                let mut indiv = [0.0; N];
                for i in 0..N {
                    indiv[i] = rng.random_range(self.generation_range.clone());
                }
                let indiv = indiv.map(|x| NotNan::new(x).unwrap());

                RealSolution {
                    inner: indiv,
                    generator: self,
                }
            })
            .collect()
    }
}
