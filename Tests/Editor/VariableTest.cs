using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using UnityEngine;

namespace chainer
{
    public class VariableTest
    {
        MatrixBuilder<float> builder = Matrix<float>.Build;

        [Test]
        public void backwardが計算グラフ上を伝搬する()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
            var constant = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
            var target = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose());
            var loss = functions.MeanSquaredError.ForwardStatic(
                x + constant,
                target
            );
            Assert.IsNull(x.Grad);
            loss.Backward();
            Assert.IsNotNull(x.Grad);
        }

        /// <summary>
        ///
        /// </summary>
        /// <code>
        /// chain = chainer.Chain(fc = chainer.links.Linear(3, 1))
        /// chain.fc.W.data = numpy.array([[-1, 0, 1]], dtype=numpy.float32)
        /// chain.fc.b.data = numpy.array([1], dtype=numpy.float32)
        /// input1 = chainer.Variable(numpy.array([[[[4,3,2]]]], dtype=numpy.float32))
        /// input2 = chainer.Variable(numpy.array([[[[10,11,12]]]], dtype=numpy.float32))
        /// target = chainer.Variable(numpy.array([[100]], dtype=numpy.float32))
        /// loss = chainer.functions.mean_squared_error(chain.fc(input1), target) + chainer.functions.mean_squared_error(chain.fc(input1), target) + chainer.functions.mean_squared_error(chain.fc(input2), target)
        /// loss.data
        /// >>> array(29811.0, dtype=float32)
        /// chain.fc.W.zerograd()
        /// loss.backward()
        /// chain.fc.W.grad
        /// >>> array([[-3556., -3346., -3136.]], dtype=float32)
        /// </code>
        [Test]
        public void 合流のある計算グラフでもpythonと同じBakcwardになる()
        {
            var chain = new helper.models.VerySmallChain();
            var optimizer = new chainer.optimizers.Adam();
            var input1 = new Variable(builder.DenseOfArray(new float[,] {{4, 3, 2}}));
            var input2 = new Variable(builder.DenseOfArray(new float[,] {{10, 11, 12}}));
            var target = new Variable(builder.DenseOfArray(new float[,] {{100}}));
            optimizer.Setup(chain);
            Helper.AssertMatrixAlmostEqual(chain.fc._Params["W"].Value, builder.DenseOfArray(new float[,]{{-1, 0, 1}}));
            Helper.AssertMatrixAlmostEqual(chain.fc._Params["b"].Value, builder.DenseOfArray(new float[,]{{1}}));

            var loss = chainer.functions.MeanSquaredError.ForwardStatic(
                           chain.Forward(input1),
                           target
                       ) + chainer.functions.MeanSquaredError.ForwardStatic(
                           chain.Forward(input1),
                           target
                       ) + chainer.functions.MeanSquaredError.ForwardStatic(
                           chain.Forward(input2),
                           target
                       );
            Helper.AssertMatrixAlmostEqual(
                loss.Value,
                builder.DenseOfArray(new float[,]{{29811.0f}}),
                delta: 0.01f
            );
            loss.Backward();
            Helper.AssertMatrixAlmostEqual(
                chain.fc._Params["W"].Grad,
                builder.DenseOfArray(new float[,]{{-3556f, -3346f, -3136f}}),
                delta: 0.01f
            );

        }

    }
}