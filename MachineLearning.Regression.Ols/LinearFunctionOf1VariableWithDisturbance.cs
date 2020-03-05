namespace MachineLearning.Regression.Ols
{
    public sealed class LinearFunctionOf1VariableWithDisturbance
    {
        private readonly float _disturbance;

        public LinearFunctionOf1VariableWithDisturbance(float x, float disturbance)
        {
            X = x;
            _disturbance = disturbance;
        }

        public float Y => 2 * X + 1 + _disturbance;

        public float X { get; }
    }
}
