﻿using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.links
{
    public class Linear : Link
    {
        private readonly LinkedList<functions.Linear> _functionPool = new LinkedList<functions.Linear>();
        private readonly bool _reuseAfterBackward;

        public Linear(int inSize, int outSize, bool reuseAfterBackward = false)
        {
            _reuseAfterBackward = reuseAfterBackward;
            _Params["W"] = new Variable(Matrix<float>.Build.Random(outSize, inSize));
            _Params["b"] = new Variable(Matrix<float>.Build.Random(1, outSize));
        }

        public override Variable Forward(Variable x)
        {
            var oldFunction = _functionPool.FirstOrDefault(oldFunc => oldFunc.Reusable);
            if (oldFunction == null)
            {
                var function = new functions.Linear(reuseAfterBackward: _reuseAfterBackward);
                _functionPool.AddLast(function);
                return function.Forward(new List<Variable>() {x, _Params["W"], _Params["b"]});
            }
            else
            {
                _functionPool.Remove(oldFunction);
                var function = new functions.Linear(oldFunction);
                return function.Forward(new List<Variable>() {x, _Params["W"], _Params["b"]});
            }
        }
    }
}