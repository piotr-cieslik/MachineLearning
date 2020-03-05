using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MachineLearning.Regression.Ols
{
    public sealed class RegressionResults
    {
        private readonly string _header;
        private readonly RegressionModelParameters _modelParamters;
        private readonly RegressionMetrics _regressionMetrics;

        public RegressionResults(
            string header,
            RegressionModelParameters modelParamters,
            RegressionMetrics regressionMetrics)
        {
            _header = header;
            _modelParamters = modelParamters;
            _regressionMetrics = regressionMetrics;
        }

        public void PrintToConsole()
        {
            Console.WriteLine(_header);
            var function =
                string.Join(
                    " + ",
                    _modelParamters.Weights.Select((weight, index) => $"{weight}*x{index + 1}")) + " + " + _modelParamters.Bias;
            Console.WriteLine($"Function: {function}");
            Console.WriteLine($"MSR:\t{_regressionMetrics.MeanSquaredError}");
            Console.WriteLine("");
        }
    }
}
