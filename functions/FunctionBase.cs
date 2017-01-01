using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public abstract class FunctionBase<ImplementType>: Function where ImplementType : FunctionBase<ImplementType>, new()
    {
        public static Variable ForwardStatic(Variable v1, Variable v2)
        {
            return (new ImplementType()).Forward(new List<Variable>() {v1, v2});
        }

    }

    public abstract class Function
    {
        public List<Variable> Inputs;
        public Variable Output;

        public Variable Forward(List<Variable> inputs)
        {
            Inputs = inputs;
            Output = _forward(inputs);
            Output.SetCreator(this);
            return Output;
        }

        public Variable Forward(Variable x0)
        {
            return Forward(new List<Variable>() {x0});
        }

        public Variable Forward(Variable x0, Variable x1)
        {
            return Forward(new List<Variable>() {x0, x1});
        }

        public List<Matrix<float>> Backward(Matrix<float> gy)
        {
            return _backward(Inputs.Select(x => x.Value).ToList(), gy);
        }

        protected abstract Variable _forward(List<Variable> inputs);
        protected abstract List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy);
    }
}