class Context:
    def __init__(self):
        self.saved_objects = []

    def save_for_backward(self, *args):
        self.saved_objects = args


class Function:
    @classmethod
    def apply(cls, *args):
        context = Context()
        function_output = cls.forward(context, *args)

        # Create output obj with the same type as input
        requires_grad = any(getattr(obj, "requires_grad", False) for obj in args)
        output_obj = type(args[0])(function_output, requires_grad)

        # Setup output object
        if requires_grad:
            function = cls()
            function.context = context
            function.parents = args
            output_obj.grad_fn = function

        return output_obj

    @staticmethod
    def forward(context, *args):
        raise NotImplementedError

    @staticmethod
    def backward(context, grad_output):
        raise NotImplementedError


class Variable:
    def __init__(self, child_class, requires_grad=False):
        self.grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad

        self.child_class = child_class

    def accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def backward(self, retain_graph = False):
        self.grad = self.child_class(1.0) if self.grad is None else self.grad

        topology = self.__build_topology(self)

        for variable in reversed(topology):
            grad = variable.grad_fn.backward(variable.grad_fn.context, variable.grad)

            if not isinstance(grad, tuple):
                grad = (grad, )
            for parent, g in zip(variable.grad_fn.parents, grad):
                if parent.requires_grad:
                    parent.accumulate_grad(g)

            if not retain_graph:
                variable.grad_fn = None

    def __build_topology(self, variable, visited=None, topology=None):
        if visited is None:
            visited = set()
        if topology is None:
            topology = []

        if variable not in visited and variable.grad_fn is not None:
            visited.add(variable)
            for parent in variable.grad_fn.parents:
                self.__build_topology(parent, visited, topology)
            topology.append(variable)

        return topology