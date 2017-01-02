using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class Sigmoid : FunctionBase<Sigmoid>
    {
        protected override Variable _forward(List<Variable> inputs)
        {
            var input = inputs[0].Value.Clone();
            var result = input.Map(x => (Math.Tanh(x*0.5) + 1) * 0.5).ToSingle();
            return new Variable(result);
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            throw new System.NotImplementedException();
        }
    }
}