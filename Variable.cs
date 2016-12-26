using MathNet.Numerics.LinearAlgebra;

namespace chainer
{
    public class Variable
    {
        public Matrix<float> Value;
        public Matrix<float> Grad;

        public Variable(Matrix<float> value)
        {
            Value = value;
            Grad = Matrix<float>.Build.Dense(value.RowCount, value.ColumnCount, 0f);
        }
    }
}