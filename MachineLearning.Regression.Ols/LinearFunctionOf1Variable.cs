namespace MachineLearning.Regression.Ols
{
    public sealed class LinearFunctionOf1Variable
    {
        public LinearFunctionOf1Variable(float x)
        {
            X = x;
        }

        public float Y => 2 * X + 1;

        public float X { get; }
    }
}
