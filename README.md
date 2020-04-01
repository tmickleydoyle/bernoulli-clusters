[![Latest Release](https://img.shields.io/github/release/tmickleydoyle/bernoulli-clusters.svg)](https://github.com/tmickleydoyle/bernoulli-clusters/releases)
[![Build Status](https://github.com/tmickleydoyle/bernoulli-clusters/workflows/build/badge.svg)](https://github.com/tmickleydoyle/bernoulli-clusters/actions)
[![Coverage Status](https://coveralls.io/repos/github/tmickleydoyle/bernoulli-clusters/badge.svg?branch=master)](https://coveralls.io/github/tmickleydoyle/bernoulli-clusters?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/tmickleydoyle/bernoulli-clusters)](https://goreportcard.com/report/github.com/tmickleydoyle/bernoulli-clusters)

# bernoulli-clusters
Bernoulli Mixture Model for Sparse Data


Example:

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/tmickleydoyle/bernoulli-clusters/bmm"
)

func main() {
	sampleData := sampleArray(3, 2)
	sampleClusters := 2

	fmt.Println(sampleData)

	m := new(bmm.Model)

	m.Fit(sampleData, sampleClusters)

	predictData := []int{1}

	prob := m.Predict(predictData)

	fmt.Printf("Predict Cluster Probability: %G\n", prob)
}

func sampleArray(x int, y int) [][]int {
	a := make([][]int, x)

	for i := 0; i < x; i++ {
		a[i] = make([]int, y)
	}

	for i, row := range a {
		for ni, v := range row {
			if v >= 0 {
				randNum := rand.Intn(y)
				a[i][ni] = randNum
			}
		}
	}
	return a
}
```
