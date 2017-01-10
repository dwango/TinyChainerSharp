using System.Collections.Generic;
using chainer.functions;
using MathNet.Numerics.LinearAlgebra;


namespace chainer
{
    public class SimpleChain : Chain
    {
        public SimpleChain() : base(new Dictionary<string, Variable>()
        {
            {"const", new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose())}
        })
        {
        }

        public Variable Forward(Variable x)
        {
            Variable result;
            result = Params["const"] + x;
            return result;
        }
    }

    public class ChainTest
    {
        [NUnit.Framework.Test]
        public void ChainのParameterがoptimizerで更新される()
        {
            var optimizer = new SGD(lr: 0.001f);
            var chain = new SimpleChain();
            optimizer.Setup(chain);

            var loss = MeanSquaredError.ForwardStatic(
                chain.Forward(new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose())),
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            );

            var before = chain.Params["const"].Value;
            optimizer.ZeroGrads();
            loss.Backward();
            optimizer.Update();
            var after = chain.Params["const"].Value;
            Helper.MatrixNotAlmostEqual(before, after, delta: 0);
        }
    }
}