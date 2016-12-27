using System.Collections.Generic;
using System.Linq;

namespace chainer.functions
{
    public class Add : FunctionBase
    {
        public static Variable add(Variable v1, Variable v2)
        {
            return (new Add()).Forward(new List<Variable>(){v1, v2});
        }

        protected override Variable _forward(IEnumerable<Variable> inputs)
        {
            var inputList = inputs.ToList();
            return new Variable(inputList[0].Value + inputList[1].Value);
        }
    }
}