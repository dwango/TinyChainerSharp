using System;
using System.Collections.Generic;
using System.Linq;
using chainer.helper.models;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer.optimizers
{

    public class AdamTest
    {
        MatrixBuilder<float> builder = Matrix<float>.Build;

        public void AssertConvergeAfterTraining(Matrix<float>[,] data)
        {
            var logic = new LogicalOperationChain();
            var optimizer = new optimizers.Adam(alpha: 0.1f);
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
//                    UnityEngine.Debug.Log($"loss[{epoch}]: {loss.Value}");
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


    }
}