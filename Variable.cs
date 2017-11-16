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
        public Matrix<float> CurrentGrad = null;
        private Matrix<float> AddGradBuf = null;
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

        public void ClearGrad()
        {
            Grad = null;
            CurrentGrad = null;
        }

        /// <summary>
        /// Same as inputs[i].Grad += grad, but memory efficient
        /// </summary>
        public void AddGrad(Matrix<float> grad)
        {
            if (AddGradBuf == null)
            {
                AddGradBuf = Grad.Clone();
            }
            Grad.Add(grad, AddGradBuf);
            Grad = AddGradBuf;
        }

        public void Backward()
        {
            if (CurrentGrad == null)
            {
                Grad = CurrentGrad =
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
                var input_grads = targetFunction.Backward(output);
                for (int i = 0; i < inputs.Count; i++)
                {
                    inputs[i].CurrentGrad = input_grads[i]; // backward用
                    if (inputs[i].Grad == null)
                    {
                        inputs[i].Grad = input_grads[i];
                    }
                    else
                    {
                        inputs[i].AddGrad(input_grads[i]);
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