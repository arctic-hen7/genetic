use crate::population::{Individual, Population};
use generic_array::GenericArray;

/// A trait for systems that "mutate" an individual in a population. In a genetic algorithm, these
/// are used to ensure diversity remains in the population, allowing the algorithm to explore the
/// search space of possible solutions.
///
/// Mutators do not directly mutate individuals, but rather produce a pattern of
/// [`ParameterMutation`]s based on the [`Parameter`]s the individuals have. This allows users to
/// easily swap out mutation strategies without modifying their own mutation handling logic. In
/// essence, a mutator is responsible for producing a pattern of perturbations to each parameter an
/// individual has, and then [`Individual::mutate`] is responsible for applying that pattern of
/// changes.
pub trait Mutator<I: Individual> {
    /// Produces a mutation shape for the individual, given their shape (same as
    /// [`Individual::shape`], just pre-computed).
    fn mutate(
        &self,
        individual_shape: &GenericArray<Parameter, I::N>,
    ) -> GenericArray<ParameterMutation, I::N>;

    /// This method is called at the *end* of each generation, to allow a mutator to track
    /// cross-generational statistics. It is provided with the new population, along with the
    /// previous generation.
    ///
    /// All members of the previous generation will still have their fitnesses tracked by the new
    /// [`Population`], so [`Population::fitness`] can be used cheaply to get their fitnesses
    /// (without needing any additional computation, so parallelism is *not* necessary).
    /// Additionally, the old individuals will be sorted *from worst fitness to best*.
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>);
}

/// The types of parameters an individual can have from the perspective of mutation. These allow
/// specifying a shape that tells the mutator what kind of perturbation to produce.
///
/// Because there are often many mutation strategies, and fundamentally they only depend on a
/// knowledge of the parameters an individual has, if an individual can be represented as a set of
/// parameters (e.g. real, bool, real), then we can use any mutator to generate some kind of
/// perturbation for it (e.g. -1.5%, flip, +0.7%), which can then be applied in an individual-
/// specific way. This allows for more flexible mutation strategies and for them to be swapped out
/// quickly in an algorithm to compare performance.
pub enum Parameter {
    /// A real number, which will be mutated with a real-valued percentage change (e.g. `-0.15`
    /// indicate a -15% change).
    Real,
    /// A boolean, which will be mutated with a toggle (i.e. `true` means flip, `false` means leave
    /// as whatever it is).
    Bool,
}

/// The mutation of a [`Parameter`].
pub enum ParameterMutation {
    /// A real number, which will be mutated with a real-valued percentage change (e.g. `-0.15`
    /// indicate a -15% change).
    Real(f64),
    /// A boolean, which will be mutated with a toggle (i.e. `true` means flip, `false` means leave
    /// as whatever it is).
    Bool(bool),
}

/// A mutator that does nothing. Because mutators are mandatory, if this is used in a final
/// simulation, an error will occur.
pub(crate) struct NullMutator;
impl<I: Individual> Mutator<I> for NullMutator {
    fn mutate(&self, _: &GenericArray<Parameter, I::N>) -> GenericArray<ParameterMutation, I::N> {
        unimplemented!()
    }
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {
        unimplemented!()
    }
}
