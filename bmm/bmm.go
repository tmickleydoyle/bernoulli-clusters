package bmm

import (
	"errors"
	"log"
	"math"
	"math/rand"
	"time"
)

// Model represents a probabilistic clustering model using Bernoulli Mixture Model (BMM).
// The model includes parameters for evaluating and predicting clusters for discrete data.
type Model struct {
	Pi       []float64   // Cluster weights
	Mu       [][]float64 // Probability distributions for each cluster and feature
	Z        [][]float64 // Posterior probabilities (responsibilities) for each data point
	Clusters int         // Number of clusters
}

// NewModel initializes and returns a new instance of Model.
func NewModel(clusters int, features int) (*Model, error) {
	if clusters <= 0 || features <= 0 {
		return nil, errors.New("clusters and features must be greater than zero")
	}
	
	// Seed for reproducibility in randomness.
	rand.Seed(time.Now().UnixNano())
	
	// Initialize model parameters.
	model := &Model{
		Pi:       initializePi(clusters),
		Mu:       initializeMu(clusters, features, randomize=true),
		Z:        initializeZ(clusters, features),
		Clusters: clusters,
	}
	return model, nil
}

// initializePi returns a slice with equal probabilities for each cluster.
func initializePi(clusters int) []float64 {
	pi := make([]float64, clusters)
	equalProb := 1.0 / float64(clusters)
	for i := range pi {
		pi[i] = equalProb
	}
	return pi
}

// initializeMu returns a nested slice for Mu, with random values if `randomize` is true.
func initializeMu(clusters, features int, randomize bool) [][]float64 {
	mu := make([][]float64, clusters)
	for i := range mu {
		mu[i] = make([]float64, features)
		for j := range mu[i] {
			if randomize {
				mu[i][j] = rand.Float64()
			} else {
				mu[i][j] = 1.0 / float64(features)
			}
		}
	}
	return mu
}

// initializeZ returns a slice initialized with equal probabilities for each cluster.
func initializeZ(dataPoints, clusters int) [][]float64 {
	z := make([][]float64, dataPoints)
	for i := range z {
		z[i] = make([]float64, clusters)
		for j := range z[i] {
			z[i][j] = 1.0 / float64(clusters)
		}
	}
	return z
}

// Fit trains the model with the given data and number of clusters.
// `data` is expected to be a slice of integer slices, where each inner slice represents a data point with feature indices.
func (m *Model) Fit(data [][]int) error {
	// Validate input data
	if len(data) == 0 || len(data[0]) == 0 {
		return errors.New("data cannot be empty")
	}
	
	// Preprocess data: remove duplicates and validate feature indices.
	processedData, err := preprocessData(data, m.featureCount())
	if err != nil {
		return err
	}

	// Expectation-Maximization (EM) loop
	for change := true; change; {
		change = m.expectationStep(processedData)
		change = m.maximizationStep(processedData) || change
	}
	
	m.normalizePi()
	return nil
}

// featureCount returns the number of features based on the length of Mu's inner slice.
func (m *Model) featureCount() int {
	if len(m.Mu) == 0 {
		return 0
	}
	return len(m.Mu[0])
}

// preprocessData removes duplicates from data points and checks for feature index validity.
func preprocessData(data [][]int, featureCount int) ([][]int, error) {
	processed := make([][]int, len(data))
	for i, row := range data {
		uniqueFeatures := unique(row)
		for _, feature := range uniqueFeatures {
			if feature < 0 || feature >= featureCount {
				return nil, errors.New("data contains invalid feature index")
			}
		}
		processed[i] = uniqueFeatures
	}
	return processed, nil
}

// unique removes duplicate values from an integer slice.
func unique(slice []int) []int {
	uniqueSet := map[int]bool{}
	result := []int{}
	for _, val := range slice {
		if !uniqueSet[val] {
			uniqueSet[val] = true
			result = append(result, val)
		}
	}
	return result
}

// expectationStep updates the Z matrix with the latest posterior probabilities for each data point.
func (m *Model) expectationStep(data [][]int) bool {
	change := false
	for i, row := range data {
		logProbs := make([]float64, m.Clusters)
		maxLogProb := math.Inf(-1)

		for k := 0; k < m.Clusters; k++ {
			logProbs[k] = m.logProbability(row, k)
			if logProbs[k] > maxLogProb {
				maxLogProb = logProbs[k]
			}
		}

		// Convert log-probs to probabilities with normalization.
		sumProb := 0.0
		for k := range logProbs {
			m.Z[i][k] = math.Exp(logProbs[k] - maxLogProb)
			sumProb += m.Z[i][k]
		}

		for k := range m.Z[i] {
			oldValue := m.Z[i][k]
			m.Z[i][k] /= sumProb
			if !change && math.Abs(oldValue-m.Z[i][k]) > 1e-5 {
				change = true
			}
		}
	}
	return change
}

// maximizationStep updates Pi and Mu based on the current values of Z and data.
func (m *Model) maximizationStep(data [][]int) bool {
	change := false

	// Initialize new Pi and Mu.
	newPi := make([]float64, m.Clusters)
	newMu := initializeMu(m.Clusters, m.featureCount(), randomize=false)
	nM := make([]float64, m.Clusters)

	for k := 0; k < m.Clusters; k++ {
		for i, row := range data {
			nM[k] += m.Z[i][k]
			for _, feature := range row {
				newMu[k][feature] += m.Z[i][k]
			}
		}

		// Normalize Mu for cluster k
		for d := 0; d < m.featureCount(); d++ {
			newMu[k][d] /= nM[k]
			if math.Abs(m.Mu[k][d]-newMu[k][d]) > 1e-5 {
				change = true
			}
			m.Mu[k][d] = newMu[k][d]
		}
		newPi[k] = nM[k] / float64(len(data))
	}

	m.Pi = newPi
	return change
}

// normalizePi ensures Pi sums to 1.0 across clusters.
func (m *Model) normalizePi() {
	total := 0.0
	for _, weight := range m.Pi {
		total += weight
	}
	for i := range m.Pi {
		m.Pi[i] /= total
	}
}

// logProbability calculates the log-probability of a data point belonging to a cluster.
func (m *Model) logProbability(data []int, cluster int) float64 {
	logProb := math.Log(m.Pi[cluster])
	for _, feature := range data {
		logProb += math.Log(m.Mu[cluster][feature])
	}
	return logProb
}

// Predict calculates the probability of the data point belonging to each cluster based on trained parameters.
func (m *Model) Predict(data []int) float64 {
	logProbSum := math.Inf(-1)
	for k := 0; k < m.Clusters; k++ {
		clusterLogProb := math.Log(m.Pi[k])
		for _, feature := range data {
			clusterLogProb += math.Log(m.Mu[k][feature])
		}
		logProbSum = math.LogSumExp(logProbSum, clusterLogProb)
	}
	return math.Exp(logProbSum)
}

// LogSumExp computes the log-sum-exp trick for numerical stability when summing probabilities in log-space.
func LogSumExp(a, b float64) float64 {
	if a > b {
		return a + math.Log1p(math.Exp(b-a))
	}
	return b + math.Log1p(math.Exp(a-b))
}
