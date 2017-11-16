using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class Linear : FunctionBase<Linear>
    {
        private Matrix<float> _bufferForXW = null;


        protected override Variable _forward(List<Variable> inputs)
        {
            if (inputs.Count != 3)
            {
                throw new ArgumentException("function Linear requires 3 inputs");
            }
            var x = inputs[0].Value;
            var W = inputs[1].Value;
            var b = inputs[2].Value;

            if (_bufferForXW == null)
            {
                _bufferForXW = x.TransposeAndMultiply(W);
            }
            else
            {
                x.TransposeAndMultiply(W, _bufferForXW);
            }

            return new Variable(_bufferForXW + b);
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            var x = inputs[0];
            var W = inputs[1];
            var gx = gy * W;
            var gW = gy.TransposeThisAndMultiply(x);
            var gb = gy.ColumnSums().ToColumnMatrix().Transpose();
            return new List<Matrix<float>>() {gx, gW, gb};
        }
    }
}