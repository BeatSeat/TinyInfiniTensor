#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    if (inputs.size() == 2) {
        for (size_t i = 0 ; i < inputs.size(); ++i) {
            if (inputs[i] -> size() == 0) {
                return {{inputs[1-i] -> getDims()}};
            }
        }
    }
    ShapeElem n = dims.at(dim);
    for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
        auto iDim = (*it) -> getDims();
        for (auto i = 0; i < (int)rank; ++i) {
            if (i == dim) {
                n += iDim.at(dim);
            } else {
                IT_ASSERT(iDim.at(i) == dims.at(i));
            }
        }
    }
    dims[dim] = n;
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
