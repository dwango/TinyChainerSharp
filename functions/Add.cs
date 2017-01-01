using System.Collections.Generic;

using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class Add : FunctionBase<Add>
    {
        public static Variable add(Variable v1, Variable v2)
        {
            return (new Add()).Forward(new List<Variable>(){v1, v2});
        }

        protected override Variable _forward(List<Variable> inputs)
        {
            return new Variable(inputs[0].Value + inputs[1].Value);
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            return new List<Matrix<float>>()
            {
                gy,
                gy
            };
        }
    }
}
