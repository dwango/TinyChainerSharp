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
            UnityEngine.Assertions.Assert.IsTrue(_functionPool.Count < 100000, "Something wrong with functionpool");
            var oldFunction = _functionPool.FirstOrDefault(oldFunc => oldFunc.Reusable);
            functions.Linear function;
            if (oldFunction == null)
            {
                function = new functions.Linear(reuseAfterBackward: _reuseAfterBackward);
            }
            else
            {
                _functionPool.Remove(oldFunction);
                function = new functions.Linear(oldFunction);
            }

            if (_reuseAfterBackward)
            {
                _functionPool.AddLast(function);
            }

            return function.Forward(new List<Variable>() {x, _Params["W"], _Params["b"]});
        }
    }
}