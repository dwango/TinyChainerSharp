using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using UnityEditor;

namespace chainer.links
{
    internal class LogicalOperationChain : Chain
    {
        public LogicalOperationChain(
        ) : base(new Dictionary<string, Link>()
        {
            {"fc1", new links.Linear(2, 6)},
            {"fc2", new links.Linear(6, 1)}
        })
        {
            // seed固定
            Children["fc1"]._Params["W"].Value = Matrix<float>.Build.Random(6, 2, seed: 0);
            Children["fc1"]._Params["b"].Value = Matrix<float>.Build.Random(1, 6, seed: 0);
            Children["fc2"]._Params["W"].Value = Matrix<float>.Build.Random(1, 6, seed: 1);
            Children["fc2"]._Params["b"].Value = Matrix<float>.Build.Random(1, 1, seed: 1);
        }

        public Variable Forward(Variable x)
        {
            var h = x;
            h = functions.Sigmoid.ForwardStatic(Children["fc1"].Forward(h));
            h = Children["fc2"].Forward(h);
            return h;
        }
    }

    public class LinearTest
    {
        MatrixBuilder<float> builder = Matrix<float>.Build;

        public void AssertConvergeAfterTraining(Matrix<float>[,] data)
        {
            var logic = new LogicalOperationChain();
            var optimizer = new optimizers.SGD(lr: 0.5f);
            optimizer.Setup(logic);

            var converge = false;
            for (int epoch = 0; epoch < 300; epoch++)
            {
                for (int i = 0; i < data.GetLength(0); i++)
                {
                    var input = new Variable(data[i, 0]);
                    var output = new Variable(data[i, 1]);
                    var loss = functions.MeanSquaredError.ForwardStatic(
                        logic.Forward(input),
                        output
                    );
                    optimizer.ZeroGrads();
                    loss.Backward();
                    optimizer.Update();
                }

                if (Enumerable.Range(0, data.GetLength(0))
                    .All((i) =>
                    {
                        var input = new Variable(data[i, 0]);
                        var output = new Variable(data[i, 1]);
                        var diff = logic.Forward(input).Value - output.Value;
                        return Math.Abs(diff[0, 0]) < 0.1f;
                    }))
                {
                    converge = true;
//                    UnityEngine.Debug.Log($"converge: {epoch}");
                    break;
                }
            }
            Assert.True(converge);
        }

        [Test]
        public void ORが学習できる()
        {
            var data = new Matrix<float>[,]
            {
                {
                    builder.DenseOfArray(new float[,] {{1, 1}}),
                    builder.DenseOfArray(new float[,] {{1}}),
                },
                {
                    builder.DenseOfArray(new float[,] {{0, 1}}),
                    builder.DenseOfArray(new float[,] {{1}}),
                },
                {
                    builder.DenseOfArray(new float[,] {{1, 0}}),
                    builder.DenseOfArray(new float[,] {{1}}),
                },
                {
                    builder.DenseOfArray(new float[,] {{0, 0}}),
                    builder.DenseOfArray(new float[,] {{0}}),
                },
            };
            AssertConvergeAfterTraining(data);
        }

        [Test]
        public void XORが学習できる()
        {
            var data = new Matrix<float>[,]
            {
                {
                    builder.DenseOfArray(new float[,] {{1, 1}}),
                    builder.DenseOfArray(new float[,] {{0}}),
                },
                {
                    builder.DenseOfArray(new float[,] {{0, 1}}),
                    builder.DenseOfArray(new float[,] {{1}}),
                },
                {
                    builder.DenseOfArray(new float[,] {{1, 0}}),
                    builder.DenseOfArray(new float[,] {{1}}),
                },
                {
                    builder.DenseOfArray(new float[,] {{0, 0}}),
                    builder.DenseOfArray(new float[,] {{0}}),
                },
            };
            AssertConvergeAfterTraining(data);
        }
    }
}