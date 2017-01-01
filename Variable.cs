using System;
using System.Collections.Generic;
using chainer.functions;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace chainer
{
    public class Variable
    {
        public Matrix<float> Value;
        public Matrix<float> Grad = null;
        private Function _creator;
        private bool _isLeaf;

        public Variable(Matrix<float> value)
        {
            Value = value;
            _creator = null;
            _isLeaf = true;
        }

        public void SetCreator(Function creator)
        {
            _creator = creator;
            _isLeaf = false;
        }

        public void Backward()
        {
            if (Grad == null)
            {
                Grad =
                    Matrix<float>.Build.Dense(
                        Value.RowCount, Value.ColumnCount, 1f); // LossのGradは1 (自分自身)
            }
            if (_isLeaf) return;

            var functionQueue = new LinkedList<Function>();
            functionQueue.AddLast(_creator);

            while (functionQueue.Count > 0)
            {
                var targetFunction = functionQueue.First.Value;
                functionQueue.RemoveFirst();

                var inputs = targetFunction.Inputs;
                var output = targetFunction.Output;
                var input_grads = _creator.Backward(output.Value);
                for (int i = 0; i < inputs.Count; i++)
                {
                    if (inputs[i].Grad == null)
                    {
                        inputs[i].Grad = input_grads[i];
                    }
                    else
                    {
                        inputs[i].Grad += input_grads[i];
                    }
                }
                foreach (var input in inputs)
                {
                    if (!input._isLeaf)
                    {
                        functionQueue.AddLast(input._creator);
                    }
                }
            }
        }

        public static Variable operator +(Variable x, Variable y)
        {
            return Add.ForwardStatic(x, y);
        }
    }
}