package bmm

import (
	"fmt"
	"testing"
)

func TestRoundThree(t *testing.T) {
	testNum := 1.4445

	roundNum := roundTo(testNum, 3)

	if roundNum != 1.445 {
		t.Errorf("Failed to roundTo for %G", testNum)
	}
}

func TestRoundOne(t *testing.T) {
	testNum := 1.4445

	roundNum := roundTo(testNum, 1)

	if roundNum != 1.4 {
		t.Errorf("Failed to roundTo for %G", testNum)
	}
}

func TestNestedArrayLength(t *testing.T) {
	data := nestedArray(2, 3)

	if len(data) != 2 {
		t.Errorf("Length of array is not equal to %v", 2)
	}
}

func TestNestedSliceLength(t *testing.T) {
	data := nestedArray(2, 3)

	if len(data[1]) != 3 {
		t.Errorf("Length of slice is not equal to %v", 3)
	}
}

func TestZArray(t *testing.T) {
	data := zArray(2, 3)

	if data[1][1] != 1.0/3.0 {
		t.Errorf("Value is not equal to %G", 1.0/3.0)
	}
}

func TestMuArrayNew(t *testing.T) {
	data := muArray(2, 3, true)

	if data[1][1] != 1.0/3.0 {
		t.Errorf("Value is not equal to %G", 1.0/3.0)
	}
}

func TestMuArrayOrig(t *testing.T) {
	data := muArray(2, 3, false)

	if data[1][1] == data[1][2] {
		t.Error("Values should not be equal.")
	}
}

func TestPiArray(t *testing.T) {
	data := piArray(2)

	if data[1] != 0.5 {
		t.Errorf("Value is not equal to %G", 0.5)
	}
}

func TestMax(t *testing.T) {
	data := [][]int{{1, 2}, {3, 4}}

	maxValue := max(data)

	if maxValue != 4 {
		t.Error("Max value should be equal to 4")
	}
}

func TestUnique(t *testing.T) {
	data := []int{1, 1, 1, 2}

	uniqueSlice := unique(data)

	if len(uniqueSlice) != 2 {
		t.Error("Slice does not have unique values")
	}

	if uniqueSlice[0] != 1 {
		t.Error("First postion is not equal to 1")
	}

	if uniqueSlice[1] != 2 {
		t.Error("Second postion is not equal to 2")
	}
}

func TestFit(t *testing.T) {
	data := [][]int{{1, 1}, {1, 1}, {1, 0}}

	m := new(Model)

	m.Fit(data, 2)

	predictData := []int{1}

	predValue := m.Predict(predictData)

	fmt.Println(predValue)

	if predValue != 1 {
		t.Error("Prediction is not equal to 0.5339")
	}
}
