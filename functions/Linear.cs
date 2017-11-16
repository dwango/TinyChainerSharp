using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine;
using UnityEngine.Assertions;

namespace chainer.functions
{
    public class Linear : FunctionBase<Linear>
    {
        private Variable _forwardBuffer = null;
        private List<Matrix<float>> _backwardBuffer = null;

        public Linear()
        {
        }

        public Linear(Linear oldFunction) : this()
        {
            _forwardBuffer = oldFunction._forwardBuffer;
            _backwardBuffer = oldFunction._backwardBuffer;
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

            return new Variable(x.TransposeAndMultiply(W) + b);
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            var x = inputs[0];
            var W = inputs[1];
            if (_backwardBuffer == null)
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
                _backwardBuffer[2] = gy.ColumnSums().ToColumnMatrix().Transpose();
            }
            return _backwardBuffer;
        }
    }
}