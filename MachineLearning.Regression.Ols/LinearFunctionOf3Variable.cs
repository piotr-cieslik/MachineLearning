namespace MachineLearning.Regression.Ols
{
    public sealed class LinearFunctionOf3Variable
    {
        public LinearFunctionOf3Variable(float x1, float x2, float x3)
        {
            X1 = x1;
            X2 = x2;
            X3 = x3;
        }

        public float X1 { get; }
                               
        public float X2 { get; }
                               
        public float X3 { get; }

        public float Y => 1 * X1 + 2 * X2 + 3 * X3 + 4;
    }
}
