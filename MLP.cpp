
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <array>

using namespace std;

//*************struct Connections***************

struct Connections
{
    double weight;
    double deltaWeight;
};

//***********************************************
class Neurons;
typedef vector<Neurons> Layer;

//*****************Class Neurons****************

class Neurons
{

private:
    static double transferFunction(double x)
    {
        // tanh(x)
        return tanh(x);
    }
    static double transferFunction_derivative(double x)
    {
        return (1 - (tanh(x) * tanh(x)));
    }
    double m_outputVal;
    vector<Connections> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
    static double learningRate_eta;
    static double momentum_alpha;

public:
    Neurons(unsigned numOutputs, unsigned myIndex)
    {
        for (unsigned c = 0; c < numOutputs; ++c)
        {
            m_outputWeights.push_back(Connections());
            m_outputWeights.back().weight = rand() / double(RAND_MAX);
        }

        m_myIndex = myIndex;
    }
    //-----------------------------------------------------------------------------

    void setOutputVal(double val)
    {
        m_outputVal = val;
    }

    //----------------------------------------------------------------------------

    double getOutputVal() const
    {
        return m_outputVal;
    }

    //-----------------------------------------------------------------------------
    void feedForward(const Layer &prevLayer)
    {
        double sum = 0.0;

        for (unsigned n = 0; n < prevLayer.size(); ++n)
        {
            sum += prevLayer[n].getOutputVal() *
                   prevLayer[n].m_outputWeights[m_myIndex].weight;
        }

        m_outputVal = Neurons::transferFunction(sum);
    }

    //-----------------------------------------------------------------------------
    void calc_OutputGradient(double target)
    {
        double delta = target - m_outputVal;
        m_gradient = delta * transferFunction_derivative(m_outputVal);
    }
    //-----------------------------------------------------------------------------
    double dowSum(const Layer &dow_nextLayer) const
    {
        double sum = 0.0;
        for (unsigned n = 0; n < dow_nextLayer.size() - 1; ++n)
        {
            sum += m_outputWeights[n].weight * dow_nextLayer[n].m_gradient;
        }
        return sum;
    }
    //-----------------------------------------------------------------------------
    void clac_HiddenGradients(const Layer &nextLayer)
    {
        double dow = dowSum(nextLayer);
        m_gradient = transferFunction_derivative(m_outputVal) * dow;
    }

    //-----------------------------------------------------------------------------
    void updateInputWeights(Layer &prevLayer)
    {
        for (unsigned n = 0; n < prevLayer.size(); ++n)
        {
            Neurons &neuron = prevLayer.at(n);
            double oldDeltaWeight = neuron.m_outputWeights.at(m_myIndex).deltaWeight;

            double newDeltaWeight = learningRate_eta * neuron.getOutputVal() * m_gradient + momentum_alpha * oldDeltaWeight;

            neuron.m_outputWeights.at(m_myIndex).deltaWeight = newDeltaWeight;
            neuron.m_outputWeights.at(m_myIndex).weight += newDeltaWeight;
        }
    }
};

double Neurons::learningRate_eta = 0.01;
double Neurons::momentum_alpha = 0.5;
//****************Class MLP**********************

class MLP
{
private:
    double m_error;
    double m_recentAverageError;
    double m_recentAverageErrorSmoothingFactor;

    vector<Layer> m_layers; //[layer][neuron]
public:
    MLP(const vector<unsigned> &topology)
    {
        unsigned numLayers = topology.size();

        for (unsigned loop_layerNum = 0; loop_layerNum < numLayers; ++loop_layerNum)
        {
            m_layers.push_back(Layer());
            unsigned numOutputs = loop_layerNum == (topology.size() - 1) ? 0 : topology[loop_layerNum + 1];

            for (unsigned loop_neuronNum = 0; loop_neuronNum <= topology[loop_layerNum]; ++loop_neuronNum)
            {
                m_layers.back().push_back(Neurons(numOutputs, loop_neuronNum));
                cout << "Made a neuron\n";
            }

            cout << "----Made Layer----\n";
            m_layers.back().back().setOutputVal(1.0);
        }
    }

    //------------------------------------------------------

    void feedForward(const array<double, 4> &inputVals)
    {

        assert(inputVals.size() == m_layers[0].size() - 1);

        for (unsigned i = 0; i < inputVals.size(); i++)
        {
            m_layers[0][i].setOutputVal(inputVals[i]);
        }

        for (unsigned l = 1; l < m_layers.size(); ++l)
        {
            Layer &prevLayer = m_layers[l - 1];
            for (unsigned n = 0; n < m_layers[l].size() - 1; ++n)
            {
                m_layers[l][n].feedForward(prevLayer);
            }
        }
    }

    //------------------------------------------------------

    void backProp(const array<double, 1> &targetVals)
    {
        Layer &outputLayer = m_layers.back();
        m_error = 0.0;

        for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
        {
            double delta = targetVals[n] - outputLayer[n].getOutputVal();
            m_error += delta * delta;
        }

        m_error /= outputLayer.size() - 1;
        m_error = sqrt(m_error);

        m_recentAverageError = (m_recentAverageError + m_recentAverageErrorSmoothingFactor + m_error) / (m_recentAverageErrorSmoothingFactor + 1);

        // Gradient decent

        for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
        {
            outputLayer[n].calc_OutputGradient(targetVals[n]);
        }

        for (unsigned backP_layerNum = m_layers.size() - 2; backP_layerNum > 0; --backP_layerNum)
        {
            Layer &hiddenLayer = m_layers[backP_layerNum];
            Layer &next_hiddenLayer = m_layers[backP_layerNum + 1];

            for (unsigned n = 0; n < hiddenLayer.size(); ++n)
            {
                hiddenLayer[n].clac_HiddenGradients(next_hiddenLayer);
            }
        }

        for (unsigned n = m_layers.size() - 1; n > 0; --n)
        {
            Layer &currentLayer = m_layers[n];
            Layer &prevLayer = m_layers[n - 1];

            for (unsigned m = 0; m < currentLayer.size() - 1; ++m)
            {
                currentLayer[m].updateInputWeights(prevLayer);
            }
        }
    }

    //-------------------------------------------------------

    void getResults(vector<double> &resultVals) const
    {
        resultVals.clear();

        for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
        {
            resultVals.push_back(m_layers.back()[n].getOutputVal());
        }
    }
};

//******************* MAIN ****************************
void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }

    cout << endl;
}
int main(int argc, char const *argv[])
{

    vector<unsigned> topology;

    topology = {4, 3, 3, 1};

    MLP myNet(topology);

    array<array<double, 4>, 2> inputVals = {{{4.3, 8.3, 1.0, 2.6}, {3.2, 2.3, 2.9, 3.3}}};
    array<array<double, 1>, 2> targetVals = {{{.002}, {0.004}}};
    vector<double> resultVals;

    for (size_t i = 0; i < 100; i++)
    {
        for (int j = 0; j < 2; ++j)

        {
            myNet.feedForward(inputVals[j]);
            myNet.getResults(resultVals);
            showVectorVals(j + "Target", resultVals);
            myNet.backProp(targetVals[j]);
        }
        cout << "\n__________________________________________________\n";
    }
cout << "\n___________________TESTTTT__________________\n";
    array<double, 4> testVals = {1100.3, 2001.3, 9001.0, 200.16};
    myNet.feedForward(testVals);
    myNet.getResults(resultVals);
    showVectorVals("Test", resultVals);

    cout << "\n__________________________________________________\n";

    return 0;
}
