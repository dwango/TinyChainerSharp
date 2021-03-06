﻿using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public abstract class FunctionBase<ImplementType> : Function
        where ImplementType : FunctionBase<ImplementType>, new()
    {
        public static Variable ForwardStatic(List<Variable> inputs)
        {
            return (new ImplementType()).Forward(inputs);
        }

        public static Variable ForwardStatic(Variable v1)
        {
            return ForwardStatic(new List<Variable>() {v1});
        }

        public static Variable ForwardStatic(Variable v1, Variable v2)
        {
            return ForwardStatic(new List<Variable>() {v1, v2});
        }

        public static Variable ForwardStatic(Variable v1, Variable v2, Variable v3)
        {
            return ForwardStatic(new List<Variable>() {v1, v2, v3});
        }
    }

    public abstract class Function
    {
        public List<Variable> Inputs;
        public Variable Output;
        public bool Reusable = false;

        public Variable Forward(List<Variable> inputs)
        {
            Inputs = inputs;
            Output = _forward(inputs);
            Output.SetCreator(this);
            return Output;
        }

        public Variable Forward(Variable x0)
        {
            return Forward(new List<Variable>() {x0});
        }

        public Variable Forward(Variable x0, Variable x1)
        {
            return Forward(new List<Variable>() {x0, x1});
        }

        public List<Matrix<float>> Backward(Variable gy)
        {
            UnityEngine.Assertions.Assert.IsFalse(Reusable, "cannnot backward after marked as reusable");
            return _backward(Inputs.Select(x => x.Value).ToList(), gy.CurrentGrad);
        }

        protected abstract Variable _forward(List<Variable> inputs);
        protected abstract List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy);
    }
}