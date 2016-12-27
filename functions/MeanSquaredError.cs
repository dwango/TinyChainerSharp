using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class MeanSquaredError: FunctionBase
    {
        protected override Variable _forward(IEnumerable<Variable> inputs)
        {
            var inputList = inputs.ToList();
            var diff = inputList[0].Value - inputList[1].Value;
            return new Variable(diff.Transpose() * diff);
        }

        protected override IEnumerable<Matrix<float>> _backward(IEnumerable<Matrix<float>> inputs, Matrix<float> gy)
        {
            var inputList = inputs.ToList();
            var diff = inputList[0] - inputList[1];
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
