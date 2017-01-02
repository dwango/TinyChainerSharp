using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class ReLU: FunctionBase<ReLU>
    {
        protected override Variable _forward(List<Variable> inputs)
        {
            throw new System.NotImplementedException();
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            throw new System.NotImplementedException();
        }
    }
}