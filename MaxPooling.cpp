#include <iostream>
#include <vector>
using namespace std;

// input:  [batch, channels, height, width]

vector<vector<int>> maxPooling(vector<vector<int>>& map)
{
    const static int kernel_size = 3;
    const static int pad = 1;
    const static int stride = 2;

    int row = map.size();
    if(row==0 || map[0].empty())
    {
        perror("Input Size Wrong!\n");
        return {};
    }
    int col = map[0].size();

    int out_row = (row-1)/stride + 1;
    int out_col = (col-1)/stride + 1;

    vector<vector<int>> res(out_row, vector<int>(out_col));
    for(int i=0; i<out_row; ++i)
    {
        int row_start = max(i*stride-1,0);
        int row_end = min(i*stride+1,row-1);

        for(int j=0; j<out_col; ++j)
        {
            int col_start = max(j*stride-1, 0);
            int col_end = min(j*stride+1, col-1);
            
            int max_val = 0;
            if(row_start==0 || row_start==row || col_start==0 || col_start==col)
                max_val = pad;

            for(int _i=row_start; _i<=row_end; _i++){
                for(int _j=col_start; _j<=col_end; _j++){
                    if(map[_i][_j]>max_val)
                        max_val=map[_i][_j];
                }
            }
            res[i][j] = max_val;
        }
    }
    return res;    
}

void show(vector<vector<int>>& m)
{
    if(m.size()==0 || m[0].size()==0)
        return;
    int row = m.size(), col = m[0].size();
    for(int i=0; i<row; ++i)
    {
        for(int j=0; j<col; ++j)
            cout << m[i][j] << "\t";
        cout << endl;
    }
}

int main()
{
    vector<vector<int>> f_map = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    vector<vector<int>> f_map1 = {{}};
    vector<vector<int>> feature = maxPooling(f_map);
    show(feature);
    return 0;
}
