using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class Linear : FunctionBase<Linear>
    {
        private Matrix<float> _buffer = null;


        protected override Variable _forward(List<Variable> inputs)
        {
            if (inputs.Count != 3)
            {
                throw new ArgumentException("function Linear requires 3 inputs");
            }
            var x = inputs[0].Value;
            var W = inputs[1].Value;
            var b = inputs[2].Value;

            Matrix<float> xw;
            if (_buffer == null)
            {
                xw = x.TransposeAndMultiply(W);
                _buffer = xw;
            }
            else
            {
                x.TransposeAndMultiply(W, _buffer);
                xw = _buffer;
            }

            return new Variable(xw + b);
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            var x = inputs[0];
            var W = inputs[1];
            var gx = gy * W;
            var gW = gy.Transpose() * x;
            var gb = gy.ColumnSums().ToColumnMatrix().Transpose();
            return new List<Matrix<float>>() {gx, gW, gb};
        }
    }
}