using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class MeanSquaredError: FunctionBase<MeanSquaredError>
    {
        protected override Variable _forward(List<Variable> inputs)
        {
            var diff = inputs[0].Value - inputs[1].Value;
            return new Variable(diff * diff.Transpose());
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            var diff = inputs[0] - inputs[1];
            var coefficient = 2.0f / diff.ColumnCount / diff.RowCount;
            var gx = coefficient * diff;
            return new List<Matrix<float>>()
            {
                gx,
                -gx
            };
        }
    }
}
