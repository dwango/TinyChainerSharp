using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.optimizers
{
    public class Adam : Optimizer
    {
        private class Param
        {
            public Matrix<float> M;
            public Matrix<float> V;

            private Param(int rows, int columns)
            {
                M = Matrix<float>.Build.Dense(rows: rows, columns: columns, value: 0);
                V = Matrix<float>.Build.Dense(rows: rows, columns: columns, value: 0);
            }

            public static Param BuildParamOf(Variable v)
            {
                return new Param(v.Value.RowCount, v.Value.ColumnCount);
            }
        }


        private readonly float _alpha;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _eps;
        private Dictionary<Variable, Param> _states;

        public Adam(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 0.000000001f)
        {
            _alpha = alpha;
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
        }

        protected override void _Setup()
        {
            _states = _link.GetParams()
                .ToDictionary(
                    eachParam => eachParam,
                    Param.BuildParamOf
                );
        }

        protected override void _Update()
        {
            var fix1 = 1.0f - Math.Pow(_beta1, _iterated_times);
            var fix2 = 1.0f - Math.Pow(_beta2, _iterated_times);
            var lr = _alpha * Math.Sqrt(fix2) / fix1;
            foreach (var param in _link.GetParams())
            {
                _states[param].M = _states[param].M + (1 - _beta1) * (param.Grad - _states[param].M);
                _states[param].V = _states[param].V + (1 - _beta2) * (param.Grad.PointwiseMultiply(param.Grad) - _states[param].V);
                param.Value -= _states[param]
                    .M.Multiply((float) lr)
                    .PointwiseDivide(_states[param].V.PointwisePower(0.5f) + _eps);
            }
        }
    }
}