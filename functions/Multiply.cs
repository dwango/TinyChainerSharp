using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class Multiply : FunctionBase<Multiply>
    {
        protected override Variable _forward(List<Variable> inputs)
        {
            return new Variable(inputs[0].Value.PointwiseMultiply(inputs[1].Value));
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            return new List<Matrix<float>>()
            {
                gy.PointwiseMultiply(inputs[1]),
                gy.PointwiseMultiply(inputs[0])
            };
        }
    }
}