#include<bits/stdc++.h>
using namespace std;
int a[26];
int main()
{
    char c;
    while(scanf("%c",&c)!=EOF)
        if(c>='A'&&c<='Z')
            a[(int)(c-'A')]++;
    int maxn=a[0];
    for(int i=1;i<26;i++)
        maxn=(a[i]>maxn)?a[i]:maxn;
    int flag=0;
    while(maxn>0)
    {
        for(int i=0;i<26;i++)
            if(a[i]==maxn)
                flag=i;
        for(int i=0;i<=flag;i++)
        {
            if(a[i]==maxn)
            {
                if(i==0)
                    printf("*");
                else
                    printf(" *");
                a[i]--;
            }
            else
            {
                if(i==0)
                    printf(" ");
                else
                    printf("  ");
            }
        }
        printf("\n");
        maxn--;
    }
    for(char i='A';i<='Z';i++)
    {
        if(i=='A')
            printf("%c",'A');
        else
            printf(" %c",i);
    }
    return 0;
}