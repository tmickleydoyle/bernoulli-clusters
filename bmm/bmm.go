package bmm

import (
	"math"
	"math/rand"
)

// Objects used to build the model.
// These objects do not go outside of this script.
var (
	z  [][]float64
	pi []float64
	mu [][]float64
)

// Model includes the object for evaluating and predicting
type Model struct {
	Pi       []float64
	Mu       [][]float64
	Clusters int
}

// round is used the floats
// Normal middle school rounds rules
func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

// roundTo can round to any decimal precision
func roundTo(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}

// nestedArray build an array of arrays.
func nestedArray(x int, y int) [][]float64 {
	a := make([][]float64, x)

	for i := 0; i < x; i++ {
		a[i] = make([]float64, y)
	}

	return a
}

// Build an array of arrays for the z object.
func zArray(n int, k int) [][]float64 {
	s := nestedArray(n, k)

	for i, row := range s {
		for ni, v := range row {
			if v >= 0 {
				s[i][ni] = float64(1) / float64(k)
			}
		}
	}

	return s
}

// Build an array of array for the mu object.
// The initial mu object uses random numbers.
// The mu object created in the while loop applies
// an equal weight to all of the indexes in the array.
func muArray(x int, y int, new bool) [][]float64 {
	s := nestedArray(x, y)

	for i, row := range s {
		for ni, v := range row {
			if v >= 0 {
				if new {
					s[i][ni] = 1.0 / float64(y)
				} else {
					s[i][ni] = rand.Float64()
				}
			}
		}
	}

	return s
}

// Make an array for pi object the length of the clusters.
func piArray(k int) []float64 {
	s := make([]float64, k)

	for i, v := range s {
		if v >= 0 {
			s[i] = float64(1) / float64(k)
		}
	}

	return s
}

// max finds the highest value in an array of nested arrays.
func max(data [][]int) int {
	max := 0

	for _, row := range data {
		for _, v := range row {
			if v > max {
				max = v
			}
		}
	}
	return int(max)
}

// unique converts a array into a set of unique values.
// It removes duplicate values.
func unique(intSlice []int) []int {
	keys := make(map[int]bool)
	list := []int{}
	for _, entry := range intSlice {
		if _, value := keys[entry]; !value {
			keys[entry] = true
			list = append(list, entry)
		}
	}
	return list
}

// probability calculates the odds of indexes being associated
// with each other.
func probability(data []int, k int) float64 {
	clusterProb := 1.0

	for i, v := range data {
		if v >= 0 {
			clusterProb *= mu[k][i]
		}
	}

	return clusterProb * pi[k]
}

// Fit trains the model.
// The outputs of the model are probabilities of the clusters,
// the weight of the clusters, and the total number of clusters.
func (m *Model) Fit(data [][]int, clusters int) {
	// nM, zX, newPi, and newMu get updated with each loops to
	// optimize for the best clustering of the data.
	var nM []float64
	var zX [][]float64
	var newPi []float64
	var newMu [][]float64

	K := clusters
	D := max(data) + 1
	N := len(data)
	z = zArray(N, K)
	pi = piArray(K)
	mu = muArray(K, D, false)

	// change controls if the model should keep training
	change := true
	sumz := 0.0

	for ind, row := range data {
		data[ind] = unique(row)
	}

	for change == true {
		change = false

		for n, row := range data {
			for k := 0; k < K; k++ {
				z[n][k] = probability(row, k)
				sumz += z[n][k]
			}
			for k := 0; k < K; k++ {
				z[n][k] /= sumz
			}
		}

		nM = make([]float64, K)
		zX = nestedArray(K, D)
		newPi = piArray(K)
		newMu = muArray(K, D, true)

		for k := 0; k < K; k++ {
			for n, row := range data {
				nM[k] += z[n][k]
				for _, v := range row {
					zX[k][v] += z[n][k] * 1
				}
			}

			for d := 0; d < D; d++ {
				newMu[k][d] = zX[k][d] / nM[k]
			}
			newPi[k] = nM[k] / float64(N)
		}

		for k := 0; k < K; k++ {
			if roundTo(pi[k], 3) != roundTo(newPi[k], 3) {
				change = true
				pi[k] = newPi[k]
			}
			for d := 0; d < D; d++ {
				if roundTo(mu[k][d], 3) != roundTo(newMu[k][d], 3) {
					change = true
					mu[k][d] = newMu[k][d]
				}
			}
		}
	}

	totalPi := 0.0

	for _, v := range pi {
		totalPi += v
	}

	// Normalize pi so that the weights equal 100%
	for k, v := range pi {
		if v >= 0 {
			pi[k] /= totalPi
		}
	}

	// Below are the parameters needed for using the Predict func.
	m.Pi = pi
	m.Mu = mu
	m.Clusters = K
}

// Predict uses the trained objects from Fit to make predictions
// on new data. The return of is a probability of the entire cluster.
func (m *Model) Predict(predictData []int) float64 {
	prob := 0.0
	clusterProb := 1.0

	for k := 0; k < m.Clusters; k++ {
		clusterProb = 1
		for _, v := range predictData {
			clusterProb *= m.Mu[k][v]
		}
		prob += clusterProb * m.Pi[k]
	}

	return prob
}
