# bernoulli-clusters
Bernoulli Mixture Model for Sparse Data


Example:

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/tmickleydoyle/bmm"
)

func main() {
	sampleData := sampleArray(3, 2)
	sampleClusters := 2

	fmt.Println(sampleData)

	m := new(bmm.Model)

	m.Fit(sampleData, sampleClusters)

	predictData := []int{1}

	prob := m.Predict(predictData)

	fmt.Printf("Cluster Probability: %G\n", prob)
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
