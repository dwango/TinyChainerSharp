using System;
using System.Collections.Generic;
using chainer.functions;
using MathNet.Numerics.LinearAlgebra;

namespace chainer
{
    public class Variable
    {
        public Matrix<float> Value;
        public Matrix<float> Grad;
        private FunctionBase _creator;
        private bool _isLeaf;

        public Variable(Matrix<float> value)
        {
            Value = value;
            Grad = Matrix<float>.Build.Dense(value.RowCount, value.ColumnCount, 0f);
            _creator = null;
            _isLeaf = true;
        }

        public void SetCreator(FunctionBase creator)
        {
            _creator = creator;
            _isLeaf = false;
        }

        public void Backward()
        {
//            if (!_isLeaf)
//            {
//                _creator.Ba
//            }
        }
    }
}