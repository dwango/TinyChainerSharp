using System.Collections.Generic;
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
            UnityEngine.Debug.Log(_functionPool.Count);
            var oldFunction = _functionPool.FirstOrDefault(oldFunc => oldFunc.Reusable);
            functions.Linear function;
            if (oldFunction == null)
            {
                function = new functions.Linear(reuseAfterBackward: _reuseAfterBackward);
            }
            else
            {
                var before = _functionPool.Count;
                _functionPool.Remove(oldFunction);
                var after = _functionPool.Count;
                UnityEngine.Debug.Log(string.Format("{0} -> {1}", before, after));
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