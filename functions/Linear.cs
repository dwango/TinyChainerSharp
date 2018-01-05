using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine.Assertions;

namespace chainer.functions
{
    public class Linear : FunctionBase<Linear>
    {
        private Variable _forwardBuffer = null;
        private Matrix<float> _xbBuffer = null;
        private List<Matrix<float>> _backwardBuffer = null;
        public readonly bool ReuseAfterBackward = false;

        public Linear()
        {
        }

        public Linear(bool reuseAfterBackward)
        {
            ReuseAfterBackward = reuseAfterBackward;
        }

        public Linear(Linear oldFunction) : this()
        {
            _forwardBuffer = oldFunction._forwardBuffer;
            _backwardBuffer = oldFunction._backwardBuffer;
            _xbBuffer = oldFunction._xbBuffer;
            ReuseAfterBackward = oldFunction.ReuseAfterBackward;
        }

        protected override Variable _forward(List<Variable> inputs)
        {
            if (inputs.Count != 3)
            {
                throw new ArgumentException("function Linear requires 3 inputs");
            }

            var x = inputs[0].Value;
            var W = inputs[1].Value;
            var b = inputs[2].Value;

            if (_xbBuffer == null)
            {
                _xbBuffer = x.TransposeAndMultiply(W);
            }
            else
            {
                x.TransposeAndMultiply(W, _xbBuffer);
            }

            if (!ReuseAfterBackward || _forwardBuffer == null)
            {
                _forwardBuffer = new Variable(_xbBuffer.Add(b));
            }
            else
            {
                _xbBuffer.Add(b, _forwardBuffer.Value);
            }

            return _forwardBuffer;
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            var x = inputs[0];
            var W = inputs[1];
            if (!ReuseAfterBackward || _backwardBuffer == null)
            {
                var gx = gy * W;
                var gW = gy.TransposeThisAndMultiply(x);
                var gb = gy.ColumnSums().ToColumnMatrix().Transpose();
                _backwardBuffer = new List<Matrix<float>>() {gx, gW, gb};
            }
            else
            {
                Assert.IsTrue(_backwardBuffer.Count == 3);
                gy.Multiply(W, _backwardBuffer[0]);
                gy.TransposeThisAndMultiply(x, _backwardBuffer[1]);
                _backwardBuffer[2] = gy.ColumnSums().ToRowMatrix();
            }

            return _backwardBuffer;
        }
    }
}