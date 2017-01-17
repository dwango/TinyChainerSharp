using System.Collections.Generic;
using chainer.functions;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;


namespace chainer
{
    public class SimpleLink : Link
    {
        public chainer.Variable constParam;

        public SimpleLink() : base(new Dictionary<string, Variable>()
        {
            {"const", new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose())}
        })
        {
            constParam = _Params["const"];
        }

        public Variable Forward(Variable x)
        {
            Variable result;
            result = _Params["const"] + x;
            return result;
        }
    }

    public class LinkTest
    {
        [NUnit.Framework.Test]
        public void LinkのParameterがoptimizerで更新される()
        {
            var optimizer = new optimizers.SGD(lr: 0.001f);
            var link = new SimpleLink();
            optimizer.Setup(link);

            var loss = MeanSquaredError.ForwardStatic(
                link.Forward(new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose())),
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            );

            var before = link.constParam.Value;
            optimizer.ZeroGrads();
            loss.Backward();
            optimizer.Update();
            var after = link.constParam.Value;
            Helper.AssertMatrixNotAlmostEqual(before, after, delta: 0);
        }

        [NUnit.Framework.Test]
        public void Iterationを回すと最適値になる()
        {
            var optimizer = new optimizers.SGD(lr: 0.05f);
            var link = new SimpleLink();
            optimizer.Setup(link);

            var loss = MeanSquaredError.ForwardStatic(
                link.Forward(new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose())),
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            );
            Assert.Greater(loss.Value[0, 0], 0.1f);
            var converge = false;
            for (int i = 0; i < 100; i++)
            {
                var lossEach = MeanSquaredError.ForwardStatic(
                    link.Forward(new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose())),
                    new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
                );
                if (lossEach.Value[0,0] < 0.1f)
                {
                    converge = true;
                    break;
                }
                optimizer.ZeroGrads();
                lossEach.Backward();
                optimizer.Update();
            }
            Assert.True(converge);
        }
    }
}